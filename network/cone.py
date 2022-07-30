import torch
import torch.nn as nn
from network.backbone import *
import torch.nn.functional as F



class CoNe(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, 
                    backbone='resnet50', dim=256, K=65536,  num_layers=1, use_bn=True, num_classes=1000, head_activation=nn.ReLU
                ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(CoNe, self).__init__()

        self.K = K
        self.encoder_q = BackBone(
                            backbone=backbone, dim=dim, num_layers=num_layers, use_bn=use_bn, 
                            head_activation=head_activation, num_classes=num_classes, is_ema=False
                        )

        self.encoder_k = BackBone(
                            backbone=backbone, dim=dim, num_layers=num_layers, use_bn=use_bn,
                            head_activation=head_activation, num_classes=num_classes, is_ema=True
                        )
        dim = self.encoder_q.dim
        self.num_classes = num_classes

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_label_prob", torch.randn(num_classes, K))
        self.register_buffer("queue_label", torch.zeros(K, dtype=torch.long))
        

    @torch.no_grad()
    def momentum_update_key_encoder(self, m):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, prob):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_label[ptr:ptr + batch_size] = labels
        prob = concat_all_gather(prob)
        self.queue_label_prob[:, ptr:ptr + batch_size] = prob.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def supcon_in(self, feature, target, args, sim_indices=None):
        feature_bank = self.queue.clone().detach()
        feature_labels = self.queue_label.clone().detach()
        sim_matrix = torch.mm(feature, feature_bank)

        
        if sim_indices is None:
            sim_indices = sim_matrix.topk(k=args.knn_k, dim=-1)[1]
    
        sim_weight =  F.softmax(sim_matrix.gather(1, sim_indices) / args.t_sup, dim=-1)
        pos_mask = torch.eq(target.unsqueeze(1), feature_labels[sim_indices])
        prob = (sim_weight * pos_mask).sum(1)
        return prob
    

    def generate_dc_target(self, feature, args):
        feature_bank = self.queue.clone().detach()
        queue_label_prob = self.queue_label_prob.clone().detach()

        sim_matrix = torch.mm(feature, feature_bank)
        
        sim_weight = F.softmax(sim_matrix / args.t_dc, dim=-1)
        aggregated_prob = sim_weight @ queue_label_prob.T
        return aggregated_prob


    def forward(self, im_q, im_k, target, args, cone=False):
        if not cone:
            with torch.no_grad():
                k, logits_k = self.encoder_k(im_k)
                onehot = torch.zeros_like(logits_k).fill_(args.ls / (self.num_classes-1)).scatter_(1, target.unsqueeze(1), 1-args.ls)
            
            norm_q, q_logits = self.encoder_q(im_q)
            supin_loss = torch.tensor(0, dtype=torch.float).cuda()
            dc_loss = torch.tensor(0, dtype=torch.float).cuda()
            fc_loss = -torch.sum(onehot.detach() * F.log_softmax(q_logits, dim=1), dim=1).mean()

        else:
            with torch.no_grad():
                k, logits_k = self.encoder_k(im_k)
                onehot = torch.zeros_like(logits_k).fill_(args.ls / (self.num_classes-1)).scatter_(1, target.unsqueeze(1), 1-args.ls)
            
            norm_q, q_logits = self.encoder_q(im_q)
            gt_value = self.supcon_in(norm_q, target, args, sim_indices=None)
            mask = (gt_value>1e-8)
            supin_loss = torch.sum(-gt_value[mask].log()) / im_q.size(0)

            with torch.no_grad():
                dc_target = self.generate_dc_target(k, args)
        
            with torch.no_grad():
                q_mask = F.softmax(q_logits, dim=1).min(dim=1)[0] > 1e-8
    
            fc_loss  = -torch.sum(onehot[q_mask].detach() * F.log_softmax(q_logits[q_mask], dim=1), dim=1).sum() / im_q.size(0)
            dc_loss = F.kl_div(F.log_softmax(q_logits[q_mask], dim=1), dc_target.detach()[q_mask], size_average=False) / im_q.size(0)
        
        self._dequeue_and_enqueue(k, target, F.softmax(logits_k, dim=1))
        return supin_loss, fc_loss, dc_loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output

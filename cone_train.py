import torch
from util.torch_dist_sum import *
from data.imagenet import *
from util.meter import *
import time
from network.cone import CoNe
from util.accuracy import accuracy
from data.augmentation import get_default_aug, get_eval_aug
import torch.nn.functional as F
from util.dist_init import dist_init
from torch.nn.parallel import DistributedDataParallel
import argparse
import math
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=23457)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--backbone', type=str, default='')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--bs', type=int, default=32)


parser.add_argument('--train_size', type=int, default=224)
parser.add_argument('--val_size', type=int, default=256)

parser.add_argument('--m', type=float, default=0.996)
parser.add_argument('--t_sup', type=float, default=0.1)
parser.add_argument('--t_dc', type=float, default=0.07)
parser.add_argument('--knn_k', type=int, default=512)
parser.add_argument('--k', type=int, default=65536)
parser.add_argument('--alpha_sup', type=float, default=1.0)
parser.add_argument('--alpha_dc', type=float, default=1.0)
parser.add_argument('--ls', type=float, default=0, help="label smooth")

parser.add_argument('--eval', default=False, action='store_true')
parser.add_argument('--sync_bn', default=False, action='store_true')
parser.add_argument('--use_fp16', default=False, action='store_true')
parser.add_argument('--use_bn', default=False, action='store_true')
parser.add_argument('--checkpoint', type=str, default='')

args = parser.parse_args()

epochs = args.epochs
warm_up = 5



def adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch):
    if epoch < warm_up:
        T = epoch * iteration_per_epoch + i
        warmup_iters = warm_up * iteration_per_epoch
        lr = base_lr  * T / warmup_iters
    else:
        min_lr = base_lr / 1000
        T = epoch - warm_up
        total_iters = epochs - warm_up
        lr = 0.5 * (1 + math.cos(1.0 * T / total_iters * math.pi)) * (base_lr - min_lr) + min_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def train(train_loader, model, local_rank, rank, optimizer, lr, epoch, scaler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    supin_losses = AverageMeter('SUP', ':.4e')
    dc_losses = AverageMeter('DC', ':.4e')
    fc_losses = AverageMeter('FC', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, supin_losses, fc_losses, dc_losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (samples, targets) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, lr, i, len(train_loader))
        # measure data loading time
        data_time.update(time.time() - end)

        samples = samples.cuda(local_rank, non_blocking=True)
        im_q = samples
        im_k = samples.clone()
        targets = targets.cuda(local_rank, non_blocking=True)

        cur_itr = i + epoch * len(train_loader)
        total_itr = epochs * len(train_loader)
        m = args.m if args.m > 0.996 else 1 - (1 - args.m) * (math.cos(math.pi * cur_itr / float(total_itr)) + 1) / 2
        model.module.momentum_update_key_encoder(m)

        with torch.cuda.amp.autocast(enabled=args.use_fp16):
            supin_loss, fc_loss, dc_loss = model(im_q, im_k, targets, args, epoch>=1)
            loss = supin_loss * args.alpha_sup + dc_loss * args.alpha_dc + fc_loss

        if not math.isfinite(dc_loss.item()):
            raise Exception("Epoch: {}, iter: {}, dc_loss is {},  stopping training".format(epoch, i, dc_loss.item()))

        if not math.isfinite(fc_loss.item()):
            raise Exception("Epoch: {}, iter: {}, fc_loss is {},  stopping training".format(epoch, i, fc_loss.item()))
        
        if not math.isfinite(supin_loss.item()):
            raise Exception("Epoch: {}, iter: {}, knn_loss is {},  stopping training".format(epoch, i, supin_loss.item()))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        supin_losses.update(supin_loss.item(), im_q.size(0))
        fc_losses.update(fc_loss.item(), im_q.size(0))
        dc_losses.update(dc_loss.item(), im_q.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0 and rank == 0:
            progress.display(i)



@torch.no_grad()
def test(test_loader, model, local_rank):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.eval()
    
    for i, (img, target) in enumerate(test_loader):
        img = img.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        logits = model.module.encoder_q(img)
    
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1.update(acc1[0], img.size(0))
        top5.update(acc5[0], img.size(0))

    sum1, cnt1, sum5, cnt5 = torch_dist_sum(local_rank, top1.sum, top1.count, top5.sum, top5.count)
    top1_acc = sum(sum1.float()) / sum(cnt1.float())
    top5_acc = sum(sum5.float()) / sum(cnt5.float())
    return top1_acc, top5_acc



def main():
    rank, local_rank, world_size = dist_init()
    batch_size = args.bs # single gpu
    num_workers = 8
    lr = args.lr * batch_size * world_size / 256

    if rank == 0:
        print(args)
    
    model = CoNe(backbone=args.backbone, dim=256, use_bn=args.use_bn, num_layers=args.layers, K=args.k)
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)


    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.wd, momentum=0.9)

    torch.backends.cudnn.benchmark = True
    train_dataset = Imagenet(mode='train', aug=get_default_aug(args.train_size))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True, persistent_workers=True)


    test_dataset = Imagenet(mode='val', aug=get_eval_aug(res=args.val_size))
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=(test_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=test_sampler, persistent_workers=True)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)

    if not os.path.exists('checkpoints') and rank == 0:
        os.makedirs('checkpoints')

    checkpoint_path = 'checkpoints/{}'.format(args.checkpoint)
    print('checkpoint_path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        checkpoint =  torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    
    if args.eval:
        top1, top5 = test(test_loader, model, local_rank)
        if rank == 0:
            print('* Acc@1 {:.3f} Acc@5 {:.3f}'.format(top1, top5))
    
    else:
        best_top1 = 0
        best_top5 = 0
        for epoch in range(start_epoch, epochs):
            train_sampler.set_epoch(epoch)
            train(train_loader, model, local_rank, rank, optimizer, lr, epoch, scaler)
            top1, top5 = test(test_loader, model, local_rank)
            best_top1 = max(best_top1, top1)
            best_top5 = max(best_top5, top5)
            if rank == 0:
                print('Epoch:{} * Acc@1 {:.3f} Acc@5 {:.3f} Best_Acc@1 {:.3f} Best_Acc@5 {:.3f}'.format(epoch, top1, top5, best_top1, best_top5))
                
                state_dict =  {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'epoch': epoch + 1
                }
                torch.save(state_dict, checkpoint_path)

if __name__ == "__main__":
    main()



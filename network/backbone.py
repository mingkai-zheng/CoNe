from network.resnet import *
import torch.nn as nn
import torch.nn.functional as F
from network.head import *


backbone_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet200': resnet200,
}

dim_dict = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
    'resnet200': 2048,
}



class BackBone(nn.Module):
    def __init__(self, 
                    backbone='resnet50', 
                    dim=256, num_layers=1, 
                    use_bn=False,
                    num_classes=1000,
                    head_activation=nn.ReLU,
                    is_ema=False,
                ):

        super().__init__()
        dim_in = dim_dict[backbone]

        self.is_ema = is_ema
        self.net = backbone_dict[backbone]()
        
        if num_layers <= 0 :
            self.head = nn.Identity()
            dim = dim_in
        else:
            self.head = ProjectionHead(
                                dim_in=dim_in,
                                dim_out=dim,
                                num_layers=num_layers,
                                use_bn=use_bn,
                                head_activation=head_activation,
                            )
            
        self.dim = dim
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        feat = self.net(x)
        embedding = self.head(feat)
        logits = self.fc(embedding)
        if not self.training and not self.is_ema:
            return logits
        return F.normalize(embedding), logits
        




class BackBoneTwohead(nn.Module):
    def __init__(self, 
                    backbone='resnet50', 
                    dim=256, num_layers=1, 
                    use_bn=False,
                    num_classes=1000,
                    head_activation=nn.ReLU,
                ):

        super().__init__()
        dim_in = dim_dict[backbone]

        self.net = backbone_dict[backbone]()        
        self.head = ProjectionHead(
                                    dim_in=dim_in,
                                    dim_out=dim,
                                    num_layers=num_layers,
                                    use_bn=use_bn,
                                    head_activation=head_activation,
                                )
        
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        feat = self.net(x)
        embedding = self.head(feat)
        logits = self.fc(feat)

        if not self.training:
            return logits
        return F.normalize(embedding), logits


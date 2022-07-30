import torch
import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, net, dim_in=2048, dim_out=1000, fix=False):
        super().__init__()
        self.net = net
        self.fc = nn.Linear(dim_in, dim_out)

        self.fix = fix
        if fix:
            for param in self.net.parameters():
                param.requires_grad = False
            self.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.fc.bias.data.zero_()

    def forward(self, x):
        if self.fix:
            with torch.no_grad():
                feat = self.net(x)
        else:
            feat = self.net(x)
        return self.fc(feat)



class ProjectionHead(nn.Module):
    def __init__(self, 
                    dim_in=2048, hidden_dim=None, 
                    dim_out=256, num_layers=1, 
                    use_bn=False, norm_layer=nn.BatchNorm1d, 
                    head_activation=nn.ReLU):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim_in * 2
        hidden_dim = max(hidden_dim, 4096)

        head = [nn.Linear(dim_in, hidden_dim)]
        if use_bn:
            head.append(norm_layer(hidden_dim))
        head.append(head_activation())
        for _ in range(1, num_layers):
            head.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                head.append(norm_layer(hidden_dim))
            head.append(head_activation())
        head.append(nn.Linear(hidden_dim, dim_out))
        self.head = nn.Sequential(*head)

    
    def forward(self, x):
        x = self.head(x)
        return x

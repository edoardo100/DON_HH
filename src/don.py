import torch
import torch.nn as nn
from .architectures import FNN, FNN_BN, FNN_LN, ResNet, myGRU, FourierFeatures
from .architectures import ResidualBlockCNN, AdaptFF
from .architectures import activation

#########################################
# DeepONet class
#########################################  
class DeepONet(nn.Module):
    def __init__(self, layers, activation_str, kernel_initializer, arc_b, arc_t, adapt_actfun=False):
        """ parameters are dictionaries """
        super().__init__()
        self.layer_b = layers["branch"]
        self.layer_t = layers["trunk"]
        self.act_b   = activation_str["branch"]
        self.act_t   = activation_str["trunk"]
        self.init_b  = kernel_initializer["branch"]
        self.init_t  = kernel_initializer["trunk"]
        self.arc_b   = arc_b
        self.arc_t   = arc_t
        self.adapt   = adapt_actfun
        
        if self.arc_b == "FNN":
            self.branch  = FNN(self.layer_b, self.act_b, self.init_b, self.adapt)
        elif self.arc_b == "FNN_BN":
            self.branch  = FNN_BN(self.layer_b, self.act_b, self.init_b, self.adapt)
        elif arc_b == "FNN_LN":
            self.branch  = FNN_LN(self.layer_b, self.act_b, self.init_b, self.adapt)
        elif self.arc_b == "ResNet":
            self.branch  = ResNet(ResidualBlockCNN,[3,3,3,3])
        elif arc_b == "GRU":
            self.branch = myGRU()
        else:
            raise NotImplementedError("Architecture for branch not implemented yet.")

        if self.arc_t == "FNN":
            self.trunk   = FNN(self.layer_t, self.act_t, self.init_t, self.adapt)
        elif self.arc_t == "FNN_BN":
            self.trunk  = FNN_BN(self.layer_t, self.act_t, self.init_t, self.adapt)
        elif arc_t == "FNN_LN":
            self.trunk  = FNN_LN(self.layer_t, self.act_t, self.init_t, self.adapt)
        elif arc_t == "FourierFeatures":
            self.mapping_size = 10 # number of fourier modes
            self.scale = 1
            self.trunk = nn.Sequential(FourierFeatures(self.scale, self.mapping_size),
                                       FNN_LN(self.layer_t, self.act_t, self.init_t, self.adapt))
        elif arc_t == "AdaptFF":
            self.mapping_size = 10 # number of fourier modes
            self.trunk = nn.Sequential(AdaptFF(self.mapping_size),
                                       FNN_LN(self.layer_t, self.act_t, self.init_t, self.adapt))
        else:
            raise NotImplementedError("Architecture for trunk not implemented yet.")
            
        # Final bias
        self.b = nn.parameter.Parameter(torch.tensor(0.0))

    def forward(self,x):
        b_in = x[0]
        if self.arc_b == "ResNet":
            b_in = b_in.unsqueeze(-1)
            b_in = b_in.permute(0,2,1)
        t_in = x[1]
        b_in = self.branch(b_in)
        # Notice that in trunk we apply the activation
        # also to the last layer
        t_in = activation(self.act_t)(self.trunk(t_in))
        out = torch.einsum("ij,kj->ik",b_in,t_in) # check with dataset
        # add bias
        out += self.b
        return out
    
#########################################
# DeepONet_md class (multidimensional)
#########################################  
class DeepONet_md(nn.Module):
    def __init__(self, dim, layers, activation_str, kernel_initializer, arc_b, arc_t, adapt_actfun=False):
        """ parameters are dictionaries """
        super().__init__()
        self.dim     = dim                  # 4 for HH
        self.layer_b = layers["branch"]
        self.layer_t = layers["trunk"]
        self.act_b   = activation_str["branch"]
        self.act_t   = activation_str["trunk"]
        self.init_b  = kernel_initializer["branch"]
        self.init_t  = kernel_initializer["trunk"]
        self.arc_b   = arc_b
        self.arc_t   = arc_t
        self.adapt   = adapt_actfun
        
        self.branch = nn.ModuleList()
        for _ in range(self.dim):
            if self.arc_b == "FNN":            
                self.branch.append(
                    FNN(self.layer_b, self.act_b, self.init_b, self.adapt)
                    )
            elif self.arc_b == "FNN_BN":
                self.branch.append(
                    FNN_BN(self.layer_b, self.act_b, self.init_b, self.adapt)
                    )
            elif arc_b == "FNN_LN":
                self.branch.append(
                    FNN_LN(self.layer_b, self.act_b, self.init_b, self.adapt)
                    )
            else:
                raise NotImplementedError("Architecture for branch not implemented yet.")

        if self.arc_t == "FNN":
            self.trunk   = FNN(self.layer_t, self.act_t, self.init_t, self.adapt)
        elif self.arc_t == "FNN_BN":
            self.trunk  = FNN_BN(self.layer_t, self.act_t, self.init_t, self.adapt)
        elif arc_t == "FNN_LN":
            self.trunk  = FNN_LN(self.layer_t, self.act_t, self.init_t, self.adapt)
        elif arc_t == "FourierFeatures":
            self.mapping_size = 10 # number of fourier modes
            self.scale = 1
            self.trunk = nn.Sequential(FourierFeatures(self.scale, self.mapping_size),
                                       FNN_LN(self.layer_t, self.act_t, self.init_t, self.adapt))
        elif arc_t == "AdaptFF":
            self.mapping_size = 10 # number of fourier modes
            self.trunk = nn.Sequential(AdaptFF(self.mapping_size),
                                       FNN_LN(self.layer_t, self.act_t, self.init_t, self.adapt))
        else:
            raise NotImplementedError("Architecture for trunk not implemented yet.")
            
        # Final bias
        self.b = []
        for _ in range(self.dim):
            self.b.append(
                nn.parameter.Parameter(torch.tensor(0.0))
            )

    def forward(self,x):
        b_in = x[0]
        t_in = x[1]
        b_out = []
        for i in range(self.dim):
            b_out.append(self.branch[i](b_in))
        # Notice that in trunk we apply the activation
        # also to the last layer
        t_in = activation(self.act_t)(self.trunk(t_in))
        out = []
        for i in range(self.dim):
            out.append(torch.einsum("ij,kj->ik",b_out[i],t_in)+self.b[i])       # check with dataset
        # add bias
        out = torch.stack(tuple(out[i] for i in range(self.dim)),dim=0)
        return out
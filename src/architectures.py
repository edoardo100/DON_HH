"""
architectures.py

author: Edoardo Centofanti

some pytorch architectures for DeepONet and similia.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn

def activation(act_fun):
    act_dict = {
        "ReLu"     : F.relu,
        "Tanh"     : F.tanh,
        "GELU"     : F.gelu,
        "Sigmoid"  : F.sigmoid,
        "Sin"      : torch.sin,
    }
    return act_dict[act_fun]
    
def initializer(initial):
    initial_dict = {
        "Glorot normal": torch.nn.init.xavier_normal_,
        "Glorot uniform": torch.nn.init.xavier_uniform_,
        "He normal": torch.nn.init.kaiming_normal_,
        "He uniform": torch.nn.init.kaiming_uniform_,
        "zeros": torch.nn.init.zeros_,
    }
    return initial_dict[initial]

#########################################
# activation functions and initializers
#########################################
class AdaptiveLinear(nn.Linear):
    """Applies a linear transformation to the input data as follows
    :math:`y = naxA^T + b`.
    More details available in Jagtap, A. D. et al. Locally adaptive
    activation functions with slope recovery for deep and
    physics-informed neural networks, Proc. R. Soc. 2020.

    Parameters
    ----------
    in_features : int
        The size of each input sample
    out_features : int 
        The size of each output sample
    bias : bool, optional
        If set to ``False``, the layer will not learn an additive bias
    adaptive_rate : float, optional
        Scalable adaptive rate parameter for activation function that
        is added layer-wise for each neuron separately. It is treated
        as learnable parameter and will be optimized using a optimizer
        of choice
    adaptive_rate_scaler : float, optional
        Fixed, pre-defined, scaling factor for adaptive activation
        functions
    """
    def __init__(self, in_features, out_features, bias=True, adaptive_rate=None, adaptive_rate_scaler=None):
        super(AdaptiveLinear, self).__init__(in_features, out_features, bias)
        self.adaptive_rate = adaptive_rate
        self.adaptive_rate_scaler = adaptive_rate_scaler
        if self.adaptive_rate:
            self.A = nn.Parameter(self.adaptive_rate * torch.ones(self.in_features))
            if not self.adaptive_rate_scaler:
                self.adaptive_rate_scaler = 10.0
            
    def forward(self, input):
        if self.adaptive_rate:
            return nn.functional.linear(self.adaptive_rate_scaler * self.A * input, self.weight, self.bias)
        return nn.functional.linear(input, self.weight, self.bias)

#########################################
# loss function
#########################################
class L2relLoss():
    """ sum of relative L^2 error """        
    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), 2, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), 2, 1)
        
        return torch.sum(diff_norms/y_norms)
    
    def __call__(self, x, y):
        return self.rel(x, y)
    
class MSE():
    """ sum of relative L^2 error """        
    def mse(self, x, y):
        num_examples = x.size()[0]
        diff = torch.square(x.reshape(num_examples,-1) - y.reshape(num_examples,-1))
        return torch.sum(diff)
    
    def __call__(self, x, y):
        return self.mse(x, y)

#########################################
# ResNet-CNN class
#########################################  
conv_2d = False 
class ResidualBlockCNN(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super().__init__()
        if conv_2d:
            self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
            self.conv2 = nn.Sequential(            
                        nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(out_channels))
        else:
            self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU())
            self.conv2 = nn.Sequential(            
                        nn.Conv1d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm1d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

#########################################
# ResNet class
#########################################
class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes): # num_classes should be G_dim
        super().__init__()
        self.inplanes = 16
        if conv_2d:
            self.conv1 = nn.Sequential(
                            nn.Conv2d(1,16,kernel_size=7,stride=2,padding=3),
                            nn.BatchNorm2d(16),
                            nn.ReLU())
            self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        else:
            self.conv1 = nn.Sequential(
                            nn.Conv1d(1,16,kernel_size=7,stride=2,padding=3),
                            nn.BatchNorm1d(16),
                            nn.ReLU())
            self.maxpool = nn.MaxPool1d(kernel_size=3,stride=2,padding=1)
        self.layer0  = self._make_layer(block,16,layers[0],stride=1)
        self.layer1  = self._make_layer(block,32,layers[1],stride=2)
        self.layer2  = self._make_layer(block,64,layers[2],stride=2)
        self.layer3  = self._make_layer(block,128,layers[3],stride=2)
        if conv_2d:
            self.avgpool = nn.AvgPool2d(7, stride=1)
        else:
            self.avgpool = nn.AvgPool1d(7, stride=1)
        self.fc      = nn.Linear(1280,num_classes)
    
    def _make_layer(self,block,planes,blocks,stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            if conv_2d:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes,planes,kernel_size=1,stride=stride),
                    nn.BatchNorm2d(planes)
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv1d(self.inplanes,planes,kernel_size=1,stride=stride),
                    nn.BatchNorm1d(planes)
                )
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x

#########################################
# FNN class
#########################################  
class FNN(nn.Module):
    def __init__(self, layer_sizes, activation_str, kernel_initializer,adapt_actfun=False):
        super().__init__()
        self.layers      = layer_sizes
        self.activation  = activation(activation_str)
        self.initializer = initializer(kernel_initializer)
        self.linears     = nn.ModuleList()
        self.adapt_rate  = None
        
        if adapt_actfun:
            self.adapt_rate = 0.1

        # Assembly the network
        for i in range(1,len(layer_sizes)):
            self.linears.append(
                AdaptiveLinear(layer_sizes[i-1],layer_sizes[i],adaptive_rate=self.adapt_rate)
            )
            # Initialize the parameters
            self.initializer(self.linears[-1].weight)
            # Initialize bias to zero
            initializer("zeros")(self.linears[-1].bias) 
    
    def forward(self,x):
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        return x
    
#########################################
# FNN_BN class
#########################################  
class FNN_BN(nn.Module):
    def __init__(self, layer_sizes, activation_str, kernel_initializer, adapt_actfun=False):
        super().__init__()
        self.layers      = layer_sizes
        self.activation  = activation(activation_str)
        self.initializer = initializer(kernel_initializer)
        self.linears     = nn.ModuleList()
        self.batch_layer = nn.ModuleList()
        self.adapt_rate  = None
        
        if adapt_actfun:
            self.adapt_rate = 0.1
            
        if activation_str.lower() == "tanh" or activation_str.lower() == "relu" or activation_str.lower() == "leaky_relu":
            gain = nn.init.calculate_gain(activation_str.lower())
        else:
            gain = 1

        # Assembly the network
        for i in range(1,len(layer_sizes)):
            self.linears.append(nn.Linear(layer_sizes[i-1],layer_sizes[i]))
            self.batch_layer.append(nn.BatchNorm1d(layer_sizes[i])) # BN
            # Initialize the parameters
            self.initializer(self.linears[-1].weight,gain)
            # Initialize bias to zero
            initializer("zeros")(self.linears[-1].bias)
            
    
    def forward(self,x):
        for i in range(len(self.linears) - 1):
            x = self.linears[i](x)
            x = self.batch_layer[i](x)  # Apply batch normalization
            x = self.activation(x)

        x = self.linears[-1](x)
        return x

#########################################
# FNN_LN class
#########################################         
class FNN_LN(nn.Module):
    def __init__(self, layers, activation_str, initialization_str):
        super().__init__()
        self.layers = layers # list with the number of neurons for each layer
        self.activation_str = activation_str
        self.initialization_str = initialization_str
        
        # linear layers
        self.linears = nn.ModuleList(
            [ nn.Linear(self.layers[i], self.layers[i+1]) 
              for i in range( len(self.layers) - 1 ) ])
        
        # batch normalization apllied in hidden layers
        self.layer_norm = nn.ModuleList(
            [ nn.LayerNorm(self.layers[i])
              for i in range(1, len(self.layers) - 2) ])

        self.linears.apply(self.param_initialization)
            
    #  Initialization for parameters
    def param_initialization(self, m):        
        if type(m) == nn.Linear:
            #### calculate gain 
            if self.activation_str == "tanh" or self.activation_str == "relu":
                gain = nn.init.calculate_gain(self.activation_str)
                a = 0
            elif self.activation_str == "leaky_relu":
                gain = nn.init.calculate_gain(self.activation_str, 0.01)
                a = 0.01
            else:
                gain = 1
                a = 0.01
            
            #### weights initialization
            if self.initialization_str == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight.data, gain = gain)
                
            elif self.initialization_str == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight.data, gain = gain)
                
            elif self.initialization_str == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight.data, 
                                               a = a, 
                                               nonlinearity = self.activation_str)
                
            elif self.initialization_str == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight.data, 
                                               a = a, 
                                               nonlinearity = self.activation_str)
            #### bias initialization
            torch.nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = activation(self.activation_str)(self.linears[0](x))
        for i in range(1, len(self.layers) - 2):
            x = activation(self.activation_str)(self.linears[i](self.layer_norm[i-1](x)))
        return self.linears[-1](x)   

#########################################
# myGRU class
#########################################
class myGRU(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.gru = nn.Sequential(
                      nn.GRU(input_size=input_size,hidden_size=256),
                      nn.GRU(input_size=256,hidden_size=128),
                   )

#########################################
# DeepONet class
#########################################  
class DeepONet(nn.Module):
    def __init__(self, layers, activation_str, kernel_initializer, arc_b, arc_t):
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
        
        if self.arc_b == "FNN":
            self.branch  = FNN(self.layer_b, self.act_b, self.init_b)
        elif self.arc_b == "FNN_BN":
            self.branch  = FNN_BN(self.layer_b, self.act_b, self.init_b)
        elif arc_b == "FNN_LN":
            self.branch  = FNN_LN(self.layer_b, self.act_b, self.init_b)
        elif self.arc_b == "ResNet":
            self.branch  = ResNet(ResidualBlockCNN,[3,3,3,3])

        if self.arc_t == "FNN":
            self.trunk   = FNN(self.layer_t, self.act_t, self.init_t)
        elif self.arc_t == "FNN_BN":
            self.trunk  = FNN_BN(self.layer_t, self.act_t, self.init_t)
        elif arc_t == "FNN_LN":
            self.trunk  = FNN_LN(self.layer_t, self.act_t, self.init_t)
            
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
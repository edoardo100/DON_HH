"""
architectures.py

author: Edoardo Centofanti

some pytorch architectures for DeepONet and similia.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn

#########################################
# Utilities
#########################################

def activation(act_fun):
    act_fun = act_fun.lower()
    act_dict = {
        "relu"     : F.relu,
        "tanh"     : F.tanh,
        "gelu"     : F.gelu,
        "sigmoid"  : F.sigmoid,
        "sin"      : lambda x: torch.sin(2*torch.pi*x),
    }
    return act_dict[act_fun]
    
def initializer(initial):
    initial = initial.lower()
    initial_dict = {
        "glorot normal": torch.nn.init.xavier_normal_,
        "glorot uniform": torch.nn.init.xavier_uniform_,
        "he normal": torch.nn.init.kaiming_normal_,
        "he uniform": torch.nn.init.kaiming_uniform_,
        "zeros": torch.nn.init.zeros_,
    }
    return initial_dict[initial]

def get_optimizer(model,lr,schedulerName,epochs,ntrain,batch_size):
    # AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-4)
    # lr policy
    scheduler = None
    if schedulerName is not None:
        if schedulerName.lower() == "steplr":
            # halved the learning rate every 100 epochs
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.5)
        elif schedulerName.lower() == "cosineannealinglr":
            # Cosine Annealing Scheduler (SGD with warm restart)
            iterations = epochs*(ntrain//batch_size)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = iterations)
        elif schedulerName.lower() == "reduceonplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95,
                                        patience=10, threshold=0.001, threshold_mode='rel', cooldown=0, 
                                        min_lr=2e-4, eps=1e-08, verbose=True) 
        else:
            raise ValueError("This scheduler has not been implemented yet.")
    else:
        schedulerName = "None"

    return optimizer, schedulerName, scheduler

#########################################
# Fourier Features
#########################################
class FourierFeatures(nn.Module):
    def __init__(self, scale, mapping_size):
        super().__init__()
        self.mapping_size = mapping_size
        if scale == 0:
            raise ValueError("Scale cannot be zero.")
        self.scale = scale
        self.B = self.scale * torch.randn((self.mapping_size, 1))

    def forward(self, x):
        # x is the set of coordinate and it is passed as a tensor (nt, 1)
        x_proj = torch.matmul((2. * torch.pi * x), self.B.T)
        inp = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
        
        return inp

#########################################
# Adaptive Fourier Features
#########################################
class AdaptFF(nn.Module):
    def __init__(self, mapping_size):
        super().__init__()
        self.mapping_size = mapping_size
        self.B = nn.Parameter(torch.randn(self.mapping_size,1))

    def forward(self, x):
        # x is the set of coordinate and it is passed as a tensor (nt, 1)
        x_proj = torch.matmul((2. * torch.pi * x), self.B.T)
        inp = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)

        return inp

#########################################
# Adaptive Linear
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
# MLP
#########################################
class MLP(nn.Module):
    """ shallow neural network """
    def __init__(self, in_channels, out_channels, mid_channels, act_fun="ReLu", arc=None):
        super(MLP, self).__init__()
        if arc == "FNO":
            self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
            self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)
        else:
            self.mlp1 = nn.Linear(in_channels, mid_channels)
            self.mlp2 = nn.Linear(mid_channels, out_channels)
        self.activation = activation(act_fun)

    def forward(self, x):
        x = self.mlp1(x)       # affine transformation
        x = self.activation(x) # activation function
        x = self.mlp2(x)       # affine transformation
        return x

#########################################
# Loss functions
#########################################
class L2relLoss():
    def __init__(self):
        self.name = "L2_rel"

    def get_name(self):
        return self.name
        
    """ sum of relative L^2 error """        
    def rel(self, x, y):
        diff_norms = torch.norm(x - y, 2, 1)
        y_norms = torch.norm(y, 2, 1)
        
        return torch.sum(diff_norms/y_norms)
    
    def __call__(self, x, y):
        return self.rel(x, y)

class L2relLossMultidim():
    """
    Relative L2 loss for multidimensional problems
    """
    def __init__(self):
        self.name = "L2_rel_md"
        self.loss = L2relLoss()

    def get_name(self):
        return self.name
    
    """
    The idea is that the multidimensional dataset has shape d x N x J
    in the HH case d is the number of functions (4: V,m,n and h), 
    N is the number of solutions (1600 for train, 400 for test) and
    J is the number of equispaced points (500)
    """
    
    def __call__(self, x, y):
        rel = torch.vmap(self.loss)
        return torch.mean(rel(x,y))

class MSE():
    def __init__(self):
        self.name = "mse"

    def get_name(self):
        return self.name
    
    """ sum of relative L^2 error """        
    def mse(self, x, y):
        diff = torch.square(x - y)
        return torch.sum(diff)
    
    def __call__(self, x, y):
        return self.mse(x, y)

class H1relLoss():
    def __init__(self):
        self.name = "H1_rel"
    
    def get_name(self):
        return self.name
    
    def rel(self, prediction, truth):
        # Ensure the inputs have the same shape
        assert prediction.shape == truth.shape, "Shape mismatch: prediction and truth must have the same shape"

        # Compute the L2 norm of the differences
        l2_norm_diff = torch.norm(prediction - truth, p=2, dim=1)

        # Compute the gradients of the prediction and truth
        grad_prediction = torch.autograd.grad(prediction, prediction, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
        grad_truth = torch.autograd.grad(truth, truth, grad_outputs=torch.ones_like(truth), create_graph=True)[0]

        # Compute the L2 norm of the gradient differences
        l2_norm_grad_diff = torch.norm(grad_prediction - grad_truth, p=2, dim=1)

        # Combine to get the H1 norm of the difference
        h1_norm_diff = torch.sqrt(l2_norm_diff**2 + l2_norm_grad_diff**2)

        # Compute the L2 norm of the truth
        l2_norm_truth = torch.norm(truth, p=2, dim=1)

        # Compute the gradient norm of the truth
        grad_truth = torch.autograd.grad(truth, truth, grad_outputs=torch.ones_like(truth), create_graph=True)[0]
        l2_norm_grad_truth = torch.norm(grad_truth, p=2, dim=1)

        # Combine to get the H1 norm of the truth
        h1_norm_truth = torch.sqrt(l2_norm_truth**2 + l2_norm_grad_truth**2)

        # Compute the relative error for each sample
        relative_error = h1_norm_diff / h1_norm_truth

        # Return the mean relative error
        return relative_error.mean()
    
    def __call__(self, x, y):
        return self.rel(x, y)

#class H1relLoss():
#    def __init__(self):
#        self.name = "H1_rel"
#    
#    def get_name(self):
#        return self.name
#    
#    """ Relative H^1 = W^{1,2} norm, in the equivalent Fourier formulation """
#    def rel(self, x, y, size_mean):
#        num_examples = x.size()[0]
#        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), 2, 1)
#        y_norms = torch.norm(y.reshape(num_examples,-1), 2, 1)
#        if size_mean:
#            return torch.mean(diff_norms/y_norms)
#        else:
#            return torch.sum(diff_norms/y_norms)
#
#    def __call__(self, x, y, beta = 1, size_mean = False):
#        n_t = x.size(1) 
#        # index
#        k = torch.cat((torch.arange(start = 0, end = n_t//2, step = 1),
#                       torch.arange(start = -n_t//2, end = 0, step = 1)), 
#                       0).reshape(1, n_t)
#        k = torch.abs(k)
#
#        # compute Fourier modes
#        x = torch.fft.fft(x, dim = 1)
#        y = torch.fft.fft(y, dim = 1)
#        
#        weight = 1 + beta*k**2 
#        weight = torch.sqrt(weight)
#        loss = self.rel(x*weight, y*weight, size_mean)
#
#        return loss

def get_loss(Loss):
    if Loss == "L2":
        myloss = L2relLoss()
    elif Loss == "L2_md":
        myloss = L2relLossMultidim()
    elif Loss == "mse":
        myloss = MSE()
    elif Loss == 'H1':
        myloss = H1relLoss()
    else:
        raise ValueError("Invalid Loss type provided.")
    return myloss

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
                            nn.BatchNorm1d(16),
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
            self.linears.append(AdaptiveLinear(layer_sizes[i-1],layer_sizes[i],adaptive_rate=self.adapt_rate))
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
    def __init__(self, layers, activation_str, initialization_str, adapt_actfun=False):
        super().__init__()
        self.layers = layers # list with the number of neurons for each layer
        self.activation_str = activation_str
        self.initialization_str = initialization_str
        self.adapt_rate  = None
        
        if adapt_actfun:
            self.adapt_rate = 0.1
        # linear layers
        self.linears = nn.ModuleList(
            [ AdaptiveLinear(self.layers[i],self.layers[i+1],adaptive_rate=self.adapt_rate) 
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
# TimeDistributed-like class, for porting of myGRU
# https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
#########################################
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

#########################################
# myGRU class
#########################################
class myGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(input_size=1,hidden_size=256,batch_first=True)
        self.gru2 = nn.GRU(input_size=256,hidden_size=128,batch_first=True)
        self.gru3 = nn.GRU(input_size=128,hidden_size=128,batch_first=True)
        self.gru4 = nn.GRU(input_size=128,hidden_size=256,batch_first=True)
        self.TimeDistributed = TimeDistributed(nn.Linear(256,1),batch_first=True)

    def forward(self,x):
        x = x.reshape(x.shape[0],x.shape[1],1)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x, _ = self.gru3(x)
        x, _ = self.gru4(x)
        x    = self.TimeDistributed(x)
        x    = torch.flatten(x,start_dim=1)
        return x

# main for testing classes and functions
if __name__=="__main__":

    a = torch.rand(2000,101)
    model = myGRU()
    out = model(a)
    print('out.shape = ', out.shape)
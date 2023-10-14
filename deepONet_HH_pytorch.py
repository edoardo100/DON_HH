# -*- coding: utf-8 -*-
"""
Learning Hodgkin-Huxley model with DeepONet
"""

import torch
from timeit import default_timer
import torch.nn.functional as F
import torch.nn as nn
import scipy.io as sio

#########################################
# default value
#########################################
mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
## Metal GPU acceleration on Mac OSX
## NOT WORKING since ComplexFloat and Float64 is not supported by the MPS backend
## It isn't worth to force the conversion since we have cuda machine for test 
#if torch.backends.mps.is_available():
#    mydevice = torch.device("mps")
#    print ("MPS acceleration is enabled")
#else:
#    print ("MPS device not found.")
torch.set_default_device(mydevice) # default tensor device
torch.set_default_dtype(torch.float32) # default tensor dtype

dataname     = "datasetHH_test1.mat"
split        = 1600
scaling      = "Default"
batch_size   = 100
input_size_b = 3
input_size_t = 1
out_size     = 50
adapt        = True
#### Optimizer parameters
epochs    = 20000
lr        = 1e-3
scheduler = "StepLR"
Loss      = "L2"

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

def activation(act_fun):
    act_dict = {
        "ReLu" : F.relu,
        "Tanh" : F.tanh,
        "GELU" : F.gelu,
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
# reading and scaling data
#########################################
def scale_data(data, min_value, max_value):
    data_min = torch.min(data)
    data_max = torch.max(data)
    # Apply the linear transformation
    scaled_data = (max_value - min_value) * (data - data_min) / (data_max - data_min) + min_value
    return data_min, data_max, scaled_data

def unscale_data(scaled_data, original_max, original_min):
    # Apply the inverse linear transformation
    unscaled_data = (scaled_data - original_min) / (original_max - original_min)
    # Map the unscaled data back to the original range
    unscaled_data = unscaled_data * (original_max - original_min) + original_min
    return unscaled_data

def gaussian_scale(data):
    mean = torch.mean(data)
    std_dev = torch.std(data)
    scaled_data = (data - mean) / std_dev
    return mean, std_dev, scaled_data

def inverse_gaussian_scale(scaled_data, original_mean, original_std_dev): 
    unscaled_data = scaled_data * original_std_dev + original_mean
    return unscaled_data

def load_dataset(dataname,split,scaling):
    d         = sio.loadmat(dataname)
    u_data    = torch.tensor(d['vs']).float()
    x_data    = torch.tensor(d['tspan']).float()
    v_data1   = (torch.tensor(d['iapps'])[:,0:2]).float() # pulse times
    v_data2   = (torch.tensor(d['iapps'])[:,2]).float()   # pulse intensities
    scale_fac = []
    if scaling == "Default":
        u_max, u_min, u_data = scale_data(u_data,0.0,1.0)
        x_max, x_min, x_data = scale_data(x_data,0.0,1.0)
        v_max1, v_min1, v_data1 = scale_data(v_data1,0.0,1.0)
        v_max2, v_min2, v_data2 = scale_data(v_data2,0.0,1.0)
        scale_fac = [u_max,u_min,x_max,x_min,v_max1,v_min1,v_max2,v_min2]
    elif scaling == "Gaussian":
        u_mean, u_std, u_data = gaussian_scale(u_data)
        x_mean, x_std, x_data = gaussian_scale(x_data)
        v_mean1, v_std1, v_data1 = gaussian_scale(v_data1)
        v_mean2, v_std2, v_data2 = gaussian_scale(v_data2)
        scale_fac = [u_mean,u_std,x_mean,x_std,v_mean1,v_std1,v_mean2,v_std2]
    elif scaling == "Mixed":
        u_mean, u_std, u_data = gaussian_scale(u_data)
        x_max, x_min, x_data = scale_data(x_data,0.0,1.0)
        v_mean1, v_std1, v_data1 = gaussian_scale(v_data1)
        v_mean2, v_std2, v_data2 = gaussian_scale(v_data2)
        scale_fac = [u_mean,u_std,x_max,x_min,v_mean1,v_std1,v_mean2,v_std2]
    v_data = torch.cat((v_data1,v_data2.reshape(-1,1)),axis=1)
    u_train, x_train, v_train = u_data[:split], x_data.t(), v_data[:split]
    u_test, x_test, v_test = u_data[split:], x_data.t(), v_data[split:]
    return scale_fac, u_train, x_train, v_train, u_test, x_test, v_test

def load_data_for_plotting(dataname,split):
    d         = sio.loadmat(dataname)
    u_data    = torch.tensor(d['vs']).float()
    x_data    = torch.tensor(d['tspan']).float()
    v_data1   = (torch.tensor(d['iapps'])[:,0:2]).float() # pulse times
    v_data2   = (torch.tensor(d['iapps'])[:,2]).float()   # pulse intensities
    domain = x_data.flatten().repeat(v_data1.shape[0], x_data.shape[0])
    v_data = torch.where((domain >= v_data1[:, [0]]) & (domain <= v_data1[:, [1]]), 1.0, 0.0)
    v_data = v_data*v_data2
    u_test, x_test, v_test = u_data[split:], x_data.t(), v_data[split:]
    return u_test, x_test, v_test
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

#########################################
# FNN class
#########################################  
class FNN(nn.Module):
    def __init__(self, layer_sizes, activation, kernel_initializer):
        super().__init__()
        self.layers      = layer_sizes
        self.activation  = activation
        self.initializer = kernel_initializer
        self.linears     = nn.ModuleList()
        self.adapt_rate  = None
        if adapt:
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
# DeepONet class
#########################################  
class DeepONet(nn.Module):
    def __init__(self, layers, activation, kernel_initializer):
        """ parameters are dictionaries """
        super().__init__()
        self.layer_b = layers["branch"]
        self.layer_t = layers["trunk"]
        self.act_b   = activation["branch"]
        self.act_t   = activation["trunk"]
        self.init_b  = kernel_initializer["branch"]
        self.init_t  = kernel_initializer["trunk"]
        self.branch  = FNN(self.layer_b,self.act_b,self.init_b)
        self.trunk   = FNN(self.layer_t,self.act_t,self.init_t)
        # Final bias
        self.b       = nn.parameter.Parameter(torch.tensor(0.0))

    def forward(self,x):
        b_in = x[0]
        t_in = x[1]
        b_in = self.branch(b_in)
        # Notice that in trunk we apply the activation
        # also to the last layer
        t_in = self.act_t(self.trunk(t_in))
        out = torch.einsum("ij,kj->ik",b_in,t_in) # check with dataset
        # add bias
        out += self.b
        return out

#########################################
#                 MAIN
#########################################
if __name__=="__main__":
    #### Network parameters
    layers = {"branch" : [input_size_b] + [50,50,50,50] + [out_size],
              "trunk"  : [input_size_t] + [50,50,50,50] + [out_size] }
    activ  = {"branch" : activation("Tanh"),
              "trunk"  : activation("Tanh")}
    init   = {"branch" : initializer("Glorot normal"),
              "trunk"  : initializer("Glorot normal")}
    #u_test, x_test, v_test = load_data_for_plotting(dataname,split)
    scale_fac, u_train, x_train, v_train, u_test, x_test, v_test = load_dataset(dataname,split,scaling)

    # batch loader
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(v_train, u_train),
                                                batch_size = batch_size)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(v_test, u_test),
                                              batch_size = batch_size) 
    
    model = DeepONet(layers,activ,init)

    # Count the parameters
    par_tot = sum(p.numel() for p in model.parameters())
    print("Total DeepONet parameters: ", par_tot)

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-4)
    # lr policy
    if scheduler == "StepLR":
        # halved the learning rate every 100 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10000, gamma = 0.5)
    if scheduler == "CosineAnnealingLR":
        # Cosine Annealing Scheduler (SGD with warm restart)
        iterations = epochs*(split//batch_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = iterations)

    # Loss function
    if Loss == "L2":
        myloss = L2relLoss()
    #elif Loss == 'H1':
    #    myloss = H1relLoss()
    t1 = default_timer()
    # Training process
    for ep in range(epochs+1):
        #### One epoch of training
        model.train()
        train_loss = 0
        for v, u in train_loader:
            v, u = v.to(mydevice), u.to(mydevice)
            
            optimizer.zero_grad() # annealing the gradient
            
            out = model.forward((v,x_train)) # compute the output
            
            # compute the loss
            if Loss == 'L2':
                loss = myloss(out.view(batch_size, -1), u.view(batch_size, -1))
            elif Loss == 'H1':
                loss = myloss(out, u)
                
            loss.backward() # automatic back propagation
            optimizer.step()
            if scheduler == "CosineAnnealingLR":
                scheduler.step()
                
            train_loss += loss.item() # update the loss function
            
        if scheduler == "StepLR":
            scheduler.step()
        
        #### Evaluate the model on the test set
        model.eval()
        test_l2 = 0.0
        #test_h1 = 0.0
        with torch.no_grad():
            for v, u in test_loader:
                v, u = v.to(mydevice), u.to(mydevice)
    
                out = model.forward((v,x_test))      
                test_l2 += L2relLoss()(out.view(batch_size, -1), u.view(batch_size, -1)).item()
                #test_h1 += H1relLoss()(out, u).item()
                
        train_loss /= split
        test_l2 /= u_test.shape[0]
        #test_h1 /= u_test.shape[0]
    
        t2 = default_timer()
        if ep%100==0:
            print('Epoch:', ep,
                  'Time:', t2-t1,
                  'Train_loss:', train_loss,
                  'Test_loss_l2:', test_l2, 
                  #'Test_loss_h1:', test_h1
                  )
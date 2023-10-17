# -*- coding: utf-8 -*-
"""
Learning Hodgkin-Huxley model with DeepONet
"""

import torch
from timeit import default_timer
import torch.nn.functional as F
import torch.nn as nn
import scipy.io as sio
# for test launcher interface
import os
import yaml
import argparse
# tensorboard and plotting
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
# imports for FNO
from functools import partial  

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


# plotting
import matplotlib.pyplot as plt

# Define command-line arguments
parser = argparse.ArgumentParser(description="Learning Hodgkin-Huxley model with DeepONet")
parser.add_argument("--config_file", type=str, default="default_params.yml", help="Path to the YAML configuration file")
args = parser.parse_args()

# Read the configuration from the specified YAML file
with open(args.config_file, "r") as config_file:
    config = yaml.safe_load(config_file)

param_file_name = os.path.splitext(args.config_file)[0]

# Now, `param_file_name` contains the name without the .json suffix
print("Test name:", param_file_name)

#########################################
# files names to be saved
#########################################
name_log_dir = 'exp_' + param_file_name
name_model = 'model_' + param_file_name 

#########################################
# DeepONet's hyperparameter
#########################################
dataname      = config["dataname"]
split         = config["split"]
batch_size    = config["batch_size"]
scaling       = config["scaling"]
weights_norm  = config["weights_norm"]
adapt_actfun  = config["adapt_actfun"]
scheduler     = config["scheduler"]
Loss          = config["Loss"]
epochs        = config["epochs"]
lr            = config["lr"]
u_dim         = config["u_dim"]
x_dim         = config["x_dim"]
G_dim         = config["G_dim"]
inner_layer_b = config["inner_layer_b"]
inner_layer_t = config["inner_layer_t"]
activation_b  = config["activation_b"]
activation_t  = config["activation_t"]
initial_b     = config["initial_b"]
initial_t     = config["initial_t"]
#### FNO parameters
fun_actFNO    = config["fun_actFNO"]
scalar        = config["scalar"]
arc           = config["arc"]
L             = config["L"]
d_v           = config["d_v"]
modes         = config["modes"]
#### Plotting parameters
ep_step  = config["ep_step"]
idx      = config["idx"]
n_idx    = len(idx)
plotting = config["plotting"] 

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
    v_data2   = (torch.tensor(d['iapps'])[:,[2]]).float()   # pulse intensities
    domain = x_data.flatten().repeat(v_data1.shape[0], x_data.shape[0])
    v_data = torch.where((domain >= v_data1[:, [0]]) & (domain <= v_data1[:, [1]]), 1.0, 0.0)
    v_data = v_data*v_data2
    u_test, x_test, v_test = u_data[split:], x_data.t(), v_data[split:]
    return u_test, x_test, v_test
#########################################
#            LOSS FUNCTION
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
#              FNN CLASS
#########################################  
class FNN(nn.Module):
    def __init__(self, layer_sizes, activation, kernel_initializer):
        super().__init__()
        self.layers      = layer_sizes
        self.activation  = activation
        self.initializer = kernel_initializer
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
#          DeepONet CLASSES
#########################################  
class DeepONetLinear(nn.Module):
    def __init__(self, layers, kernel_initializer):
        """ parameters are dictionaries """
        super().__init__()
        self.layer_b = layers["branch"]
        self.layer_t = layers["trunk"]
        self.init_b  = kernel_initializer["branch"]
        self.init_t  = kernel_initializer["trunk"]
        # we begin with Linear architecture: shallow single layer
        self.branch  = nn.Linear(self.layer_b[0],self.layer_b[1])
        self.trunk   = nn.Linear(self.layer_t[0],self.layer_t[1])
        # Initialize the parameters
        self.init_b(self.branch.weight)
        self.init_t(self.trunk.weight)
        # Initialize bias to zero
        initializer("zeros")(self.branch.bias) 
        initializer("zeros")(self.trunk.bias)

    def forward(self,x):
        b_in = x[0]
        t_in = x[1]
        b_in = self.branch(b_in)
        # Notice that in trunk we apply the activation
        # also to the last layer
        t_in = self.trunk(t_in)
        # multiplicative merger operation
        out = torch.einsum("ij,kj->ik",b_in,t_in) # check with dataset
        return out

#########################################
#             FNO CLASSES
#########################################
class MLP(nn.Module):
    """ shallow neural network """
    def __init__(self, in_channels, out_channels, mid_channels, activation="ReLu"):
        super(MLP, self).__init__()
        self.mlp1 = nn.Linear(in_channels, mid_channels)
        self.mlp2 = nn.Linear(mid_channels, out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.mlp1(x) # affine transformation
        x = activation(self.activation)(x) # activation function
        x = self.mlp2(x) # affine transformation
        return x

class FourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes, weights_norm, scalar):
        super().__init__()
        """
        1D Fourier layer. We initialize the parameter for the Fourier layer  
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.weights_norm = weights_norm
        self.scalar = scalar

        if scalar == "Complex":
            if weights_norm == 'Xavier':
                # Xavier normalization
                self.weights = nn.init.xavier_normal_(
                    nn.Parameter(torch.empty(in_channels, out_channels, self.modes, dtype=torch.cfloat)),
                    gain = 1/(self.in_channels*self.out_channels))
                
            elif weights_norm == 'Kaiming':
                # Kaiming normalization
                self.weights = torch.nn.init.kaiming_normal_(
                    nn.Parameter(torch.empty(in_channels, out_channels, self.modes, dtype=torch.cfloat)),
                    mode = 'fan_in', nonlinearity = fun_actFNO)
               
        elif scalar == "Real":
            if weights_norm == 'Xavier':
                # Xavier normalization
                self.weights = nn.init.xavier_normal_(
                    nn.Parameter(torch.empty(in_channels, out_channels, self.modes, 2, 
                        dtype = torch.float)),
                    gain = 1/(self.in_channels*self.out_channels))
                
            elif weights_norm == 'Kaiming':
                # Kaiming normalization
                self.weights = torch.nn.init.kaiming_normal_(
                    nn.Parameter(torch.empty(in_channels, out_channels, self.modes, 2, 
                        dtype = torch.float)),
                        mode = 'fan_in', nonlinearity = fun_actFNO)

    
    def mul_modes(self, modes, weights):
        """ Multiplication of the fourier modes """
        if self.scalar == "Complex":
            # (batch, in_channel, t), (in_channel, out_channel, t) -> (batch, out_channel, t)
            return torch.einsum("bix,iox->box", modes, weights)
        
        elif self.scalar == "Real":
            # (batch, in_channel, t), (in_channel, out_channel, t) -> (batch, out_channel, t)
            op = partial(torch.einsum, "bix,iox->box")
            return torch.stack([
                op(modes[..., 0], weights[..., 0]) - op(modes[..., 1], weights[..., 1]),
                op(modes[..., 1], weights[..., 0]) + op(modes[..., 0], weights[..., 1])
            ], dim=-1)
    
    def forward(self, x):
        """
        input --> FFT --> linear transform of relevant Fourier modes--> IFFT --> output

        Parameters
        ----------
        x : tensor
            pytorch tensor of dimension (nbatch)*(d_v)*(n_t)

        Returns
        -------
        x : tensor
            output pytorch tensor of dimension (nbatch)*(d_v)*(n_t)

        """
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
    
        # Multiply relevant Fourier modes
        if self.scalar == "Complex":
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, 
                                 dtype=torch.cfloat)
            out_ft[:, :, :self.modes] = self.mul_modes(x_ft[:, :, :self.modes], self.weights)
        
        elif self.scalar == "Real":
            x_ft = torch.stack([x_ft.real, x_ft.imag], dim = 3) # separate real from imag part
            
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, 2,
                                 dtype = torch.float)
            out_ft[:, :, :self.modes, :] = self.mul_modes(x_ft[:, :, :self.modes], self.weights)
            
            out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1]) # return to complex values
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

#########################################
#            FNO-DeepONet CLASS
#########################################
class FNO_DeepONet(nn.Module):
    """ 
    Fourier Neural Operator for approximate the Hodgkin-Huxley model
    We remove the lifting operator, which will be given by our DeepONetLinear
    """
    def __init__(self, layers, init_don, d_a, d_v, d_u, L, modes, act_fno):
        """ 
        L: int
           depth of FNO
            
        d_a : int
            input dimension
            
        d_v : int
            hidden dimension
            
        modes : int
            pari a k_{max, 1}
            
        d_u : int
            pari alla dimensione dello spazio in output 
            
        """
        super().__init__()      
        
        self.d_a = d_a
        self.d_v = d_v
        self.d_u = d_u
        self.L = L
        self.modes = modes
        self.activation = act_fno
        
        #### Lifting operator
        self.p = DeepONetLinear(layers, init_don)

        # classic architecture
        if arc == "Classic":
            #### Fourier operator
            self.fouriers = nn.ModuleList([
                FourierLayer(d_v, d_v, modes, weights_norm, scalar) 
                for _ in range(L) ])
            
            self.ws = nn.ModuleList([ nn.Linear(d_v, d_v) for _ in range(L) ])
            
            #### Projection operator
            self.q = nn.Linear(d_v, d_u) # output features is d_u: u(x,y)
            
        elif arc == "Zongyi":
            #### Fourier operator
            self.fouriers = nn.ModuleList([
                FourierLayer(d_v, d_v, modes, weights_norm, scalar) 
                for _ in range(L) ])
            
            self.ws = nn.ModuleList([ nn.Linear(d_v, d_v) for _ in range(L) ])           
            self.mlps = nn.ModuleList([ MLP(d_v, d_v, d_v) for _ in range(L) ])
            
            #### Projection operator
            self.q = MLP(d_v, d_u, 4*d_u)
    def forward(self, x):
        # initially x.size() = (n_samples)*(n_t)*(d_a)
        
        #### Apply lifting operator P
        x = self.p(x) # x.size() = (n_samples)*(n_t)*(d_v)
        x = x.unsqueeze(-1)
        x = x.permute(0, 2, 1) # x.size() = (n_samples)*(d_v)*(n_t)
        
        #### Fourier Layers
        for i in range(self.L):
            # classic architecture
            if arc == "Classic":
                for i in range(self.L):
                    x1 = self.fouriers[i](x)
                    x2 = self.ws[i](x.permute(0, 2, 1))
                    x = x1 + x2.permute(0, 2, 1)
                    if i < self.L - 1:
                        x = activation(self.activation)(x)
            # Zongyi Li architecture
            elif arc == "Zongyi":
                for i in range(self.L):
                    x1 = self.fouriers[i](x)
                    x1 = self.mlps[i](x1.permute(0, 2, 1))
                    x2 = self.ws[i](x.permute(0, 2, 1))
                    x = x1 + x2
                    x = x.permute(0, 2, 1)
                    if i < self.L - 1:
                        x = activation(self.activation)(x)           

        #### apply projection operator Q
        x = self.q(x.permute(0, 2, 1))
        return x.squeeze()

#########################################
#                 MAIN
#########################################
if __name__=="__main__":

    writer = SummaryWriter(log_dir = name_log_dir)
    
    #### Network parameters
    layers = {"branch" : [u_dim] + inner_layer_b + [G_dim],
              "trunk"  : [x_dim] + inner_layer_t + [G_dim] }
    init   = {"branch" : initializer(initial_b),
              "trunk"  : initializer(initial_t)}
    #u_test, x_test, v_test = load_data_for_plotting(dataname,split)
    scale_fac, u_train, x_train, v_train, u_test, x_test, v_test = load_dataset(dataname,split,scaling)

    # batch loader
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(v_train, u_train),
                                                batch_size = batch_size)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(v_test, u_test),
                                              batch_size = batch_size) 

    d_a = 1
    d_u = 1
    model = FNO_DeepONet(layers,init,d_a,d_v,d_u,L,modes,fun_actFNO)

    # Count the parameters
    par_tot = sum(p.numel() for p in model.parameters())
    print("Total DeepONet parameters: ", par_tot)
    writer.add_text("Parameters", 'Total parameters number: ' + str(par_tot), 0)

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
    # Unscaled dataset (for plotting)
    u_test_unscaled, x_unscaled, v_test_unscaled = load_data_for_plotting(dataname,split)
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
        if ep%10==0:
            print('Epoch:', ep, 'Time:', t2-t1,
                  'Train_loss:', train_loss, 'Test_loss_l2:', test_l2, 
                  #'Test_loss_h1:', test_h1
                  )

            writer.add_scalars('FDON_HH', {'Train_loss': train_loss,
                                                    'Test_loss_l2': test_l2,
                                                    #'Test_loss_h1': test_h1
                                                     }, ep)
    #########################################
    # plot data at every epoch_step
    #########################################
        if ep == 0:
            #### initial value of v
            esempio_test = v_test_unscaled[idx, :].to('cpu')
            esempio_test_pp = v_test[idx, :].to('cpu')
            
            X = np.linspace(0, 100, esempio_test.shape[1])
            fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
            fig.suptitle('Applied current (I_app)')
            ax[0].set(ylabel = 'I_app(t)')
            for i in range(n_idx):
                ax[i].plot(X, esempio_test[i])
                ax[i].set(xlabel = 't')
                ax[i].set_ylim([-0.2, 10])
                ax[i].grid()
            if plotting:
                plt.show()
            writer.add_figure('Applied current (I_app)', fig, 0)

            #### Approximate classical solution
            soluzione_test = u_test_unscaled[np.array(idx)]
            X = np.linspace(0, 100, soluzione_test.shape[1])
            fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
            fig.suptitle('Numerical approximation (V_m)')
            ax[0].set(ylabel = 'V_m (mV)')
            for i in range(n_idx):
                ax[i].plot(X, soluzione_test[i].to("cpu"))
                ax[i].set(xlabel = 't')
                ax[i].grid()
            if plotting:
                plt.show()
            writer.add_figure('Numerical approximation (V_m)', fig, 0)

        #### approximate solution with DON of HH model
        if ep % ep_step == 0:
            with torch.no_grad():  # no grad for effeciency reason
                    out_test = model((esempio_test_pp.to(mydevice),x_test.to(mydevice)))
                    out_test = out_test.to('cpu')
            if scaling == "Default":
                out_test = unscale_data(out_test.to(mydevice),scale_fac[1],scale_fac[0])
            elif scaling == "Gaussian":
                out_test = inverse_gaussian_scale(out_test.to(mydevice),scale_fac[0],scale_fac[1])
            elif scaling == "Mixed":
                out_test = inverse_gaussian_scale(out_test.to(mydevice),scale_fac[0],scale_fac[1])
            fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
            fig.suptitle('DON approximation (V_m)')
            ax[0].set(ylabel = 'V_m (mV)')
            for i in range(n_idx):
                ax[i].plot(X, out_test[i].to('cpu'))
                ax[i].set(xlabel = 't')
                ax[i].grid()
            if plotting:
                plt.show()
            writer.add_figure('DON approximation (V_m)', fig, ep)

            #### Module of the difference between classical and DON approximation
            diff = np.abs(out_test.to('cpu') - soluzione_test.to('cpu'))
            fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
            fig.suptitle('Module of the difference')
            ax[0].set(ylabel = '|V_m(mV)|')
            for i in range(n_idx):
                ax[i].plot(X, diff[i])                    
                ax[i].set(xlabel = 'x')
                ax[i].grid()
            if plotting:
                plt.show()
            writer.add_figure('Module of the difference', fig, ep)

    writer.flush() # per salvare i dati finali
    writer.close() # chiusura writer tensorboard
    torch.save(model, name_model)

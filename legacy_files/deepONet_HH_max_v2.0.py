# -*- coding: utf-8 -*-
"""
Learning Hodgkin-Huxley model with DeepONet
"""

import matplotlib.pyplot as plt
import numpy as np
# pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import multiprocessing
# library for reading data
import scipy.io
# timer for plotting
from timeit import default_timer
# for test launcher interface
import os
import yaml
import argparse

#########################################
# default value
#########################################
mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if mydevice == 'cuda':
    n_workers = torch.cuda.device_count() # number of GPUs
    print(f"Number of GPUs: {n_workers}")
if mydevice == 'cpu':
    n_workes = multiprocessing.cpu_count() # number of cpu's cores
    print(f"Number of CPU cores: {n_workers}")
## Metal GPU acceleration on Mac OSX
## NOT WORKING since ComplexFloat and Float64 is not supported by the MPS backend
## It isn't worth to force the conversion since we have cuda machine for test 
#if torch.backends.mps.is_available():
#    mydevice = torch.device("mps")
#    print ("MPS acceleration is enabled")
#else:
#    print ("MPS device not found.")
torch.set_default_device(mydevice) # default tensor device
torch.set_default_tensor_type(torch.FloatTensor) # default tensor dtype

#########################################
# hyperparameter file
#########################################
# Define command-line arguments
parser = argparse.ArgumentParser(description="Learning Hodgkin-Huxley model with DeepONet")
parser.add_argument("--config_file", type=str, default="default_params.yml", help="Path to the YAML configuration file")
# default_params_max <-- parametri di dafault (check retro-compatibility)
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
#### Dataset hyper-parameter
DataPath      = config["DataPath"]
ntot          = config["ntot"] # total tests
ntrain        = config["ntrain"] # training instances
ntest         = config["ntest"] # test instances
s             = config["s"]

#### Training hyper-parameters
retrain       = config["retrain"]
epochs        = config["epochs"]
lr            = config["lr"]
batch_size    = config["batch_size"]
Loss          = config["Loss"]
weight_decay  = config["weight_decay"]
scheduler     = config["scheduler"]
step_size     = config["step_size"]
gamma         = config["gamma"]

#### Architecture hyper-parameters
u_dim         = config["u_dim"]
x_dim         = config["x_dim"]
N_FourierF    = config["N_FourierF"]
G_dim         = config["G_dim"]
exp_G         = config["exp_G"]
inner_layer_b = config["inner_layer_b"]
inner_layer_t = config["inner_layer_t"]
activation_b  = config["activation_b"]
activation_t  = config["activation_t"]
arc_b         = config["arc_b"] 
arc_t         = config["arc_t"] 

try:
    init_b = config["init_b"] # default, kaimimng, xavier
except:
    init_b = "default"
    
try:
    init_t = config["init_t"]
except:
    init_t = "default"
    
try:
    data_norm = config["data_norm"]
except:
    data_norm = False

#### Plot and tensorboard
ep_step  = config["ep_step"]
idx      = config["idx"]
n_idx    = len(idx)
plotting = config["plotting"]
# On cluster plt.show() causes the program to freeze, since there is no graphics forwarding

#########################################
# reading data
#########################################
def MatReader(file_path):
    """
    Function for reading the file with data and convert it in pytorch tensor

    Parameters
    ----------
    file_path : string
        path .mat file that have to be read        

    Returns
    -------
    tspan: tensor 
        discretization time grid
        
    a : tensor
        evaluation of coefficient a, that is the input of HH problem
        
    u : tensor
        approximate solution of the HH problem with classical numerical method
        (for FNO we need a supervised learning framework)

    """
    data = scipy.io.loadmat(file_path)
    #### tspan
    tspan = data["tspan"]
    tspan = torch.from_numpy(tspan).float() # transform np.array in torch.tensor
    tspan = tspan.squeeze()
    tspan = tspan.to('cpu')
    #### Volts
    u = data["vs"]
    u = torch.from_numpy(u).float()
    u = u.to('cpu')
    #### iapps
    a = torch.zeros_like(u)
    iapps = data["iapps"]
    iapps = torch.from_numpy(iapps).float() # trasforma np.array in torch.tensor
    
    # Unefficient but intuitive implementation
    # for i in range(iapps.size(0)):
    #     for k in range(tspan.size(0)):
    #         if tspan[k] > iapps[i, 0] and tspan[k] < iapps[i, 1]:
    #             a[i, k] = iapps[i, 2]
    
    # Efficient implementation
    mask = (tspan.unsqueeze(0) >= iapps[:, [0]]) & (tspan.unsqueeze(0) <= iapps[:, [1]])
    a[mask] = iapps[:, [2]].repeat(1, tspan.size(0))[mask]
    a = a.to('cpu')
    
    return tspan, a, u

#########################################
# gaussian normalization
#########################################
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps = 1e-5):
        super().__init__()

        # x have shape of ntrain*n
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        std = self.std + self.eps
        mean = self.mean
        x = (x * std) + mean
        return x

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = torch.from_numpy(self.mean).to(device)
            self.std = torch.from_numpy(self.std).to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

#########################################
# activation function
#########################################
def activation(x, activation_str):
    """
    Activation function to utilize in the FNO architecture.
    The activaction function is the same in all the FNO architecture.
    """
    if activation_str in ["ReLu", "relu"]:
        return F.relu(x)
    elif activation_str in ["LeakyReLU", "leaky_relu"]:
        return torch.nn.LeakyReLU()(x)
    elif activation_str in ["Tanh", "tanh"]:
        return F.tanh(x)
    elif activation_str in ["GELU", "gelu"]:
        return F.gelu(x)
    elif activation_str in ["sigmoid", "Sigmoid"]:
        return F.Sigmoid(x)
    elif activation_str in ["Sin", "sin"]:
        return F.Sin(x)

#########################################
# FourierFeatures
#########################################
class FourierFeatures(nn.Module):

    def __init__(self, scale, mapping_size, device):
        super().__init__()
        self.mapping_size = mapping_size
        self.scale = scale
        self.B = scale * torch.randn((self.mapping_size, 2)).to(device)

    def forward(self, x):
        # x is the set of coordinate and it is passed as a tensor (nx, ny, 2)
        if self.scale != 0:
            x_proj = torch.matmul((2. * np.pi * x), self.B.T)
            inp = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
            return inp
        else:
            return x

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
    
class L2Loss():
    """ sum of L^2 error in d dimension """
    def __init__(self, d=2):
        self.d = d

    def rel(self, x, y):
        num_examples = x.size()[0]
        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)
        diff_norms = (h**(self.d/2))*torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), 2, 1)

        return torch.sum(diff_norms)

    def __call__(self, x, y):
        return self.rel(x, y)
    
class MSE():       
    """ MSE of the difference and then sum along the batch dimension """
    def mse(self, x, y):
        num_examples = x.size()[0]
        diff = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), 2, 1)
        return torch.sum(diff**2)
    
    def __call__(self, x, y):
        return self.mse(x, y)
    
class H1relLoss(object):
    """ Norma H^1 = W^{1,2} relativa, approssima con la trasformata di Fourier """
    def rel(self, x, y, size_mean):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), 2, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), 2, 1)
        if size_mean:
            return torch.mean(diff_norms/y_norms)
        else:
            return torch.sum(diff_norms/y_norms)

    def __call__(self, x, y, beta = 1, size_mean = False):
        n_t = x.size(1) 
        # index
        k = torch.cat((torch.arange(start = 0, end = n_t//2, step = 1),
                       torch.arange(start = -n_t//2, end = 0, step = 1)), 
                       0).reshape(1, n_t)
        k = torch.abs(k)

        # compute Fourier modes
        x = torch.fft.fft(x, dim = 1)
        y = torch.fft.fft(y, dim = 1)
        
        weight = 1 + beta*k**2 
        weight = torch.sqrt(weight)
        loss = self.rel(x*weight, y*weight, size_mean)

        return loss

#########################################
# FNN class
#########################################  
class FNN(nn.Module):
    def __init__(self, layers, activation_str):
        super().__init__()
        self.layers = layers # list with the number of neurons for each layer
        self.activation_str = activation_str
        self.retrain = retrain
        self.G_dim = G_dim # output dimension
        self.exp_G = exp_G # exponential for p at the output
        
        # fix the seed for retrain
        torch.manual_seed(self.retrain)
        
        # linear layers
        self.linears = nn.ModuleList(
            [nn.Linear(self.layers[i], self.layers[i+1]) 
            for i in range( len(self.layers) - 1 )])
    
    def forward(self,x):
        for linear in self.linears[:-1]:
            x = activation(linear(x), self.activation_str)
        x = self.linears[-1](x)
        return x/(self.G_dim ** self.exp_G)
    
#########################################
# FNN_BN class
#########################################         
class FNN_BN(nn.Module):

    def __init__(self, layers, activation_str, initialization_str):
        super().__init__()
        self.layers = layers # list with the number of neurons for each layer
        self.activation_str = activation_str
        self.retrain = retrain
        self.initialization_str = initialization_str
        
        # fix the seed for retrain
        torch.manual_seed(self.retrain)
        
        # linear layers
        self.linears = nn.ModuleList(
            [ nn.Linear(self.layers[i], self.layers[i+1]) 
              for i in range( len(self.layers) - 1 ) ])
        
        # batch normalization apllied in hidden layers
        self.batch_norm = nn.ModuleList(
            [ nn.BatchNorm1d(self.layers[i], device = mydevice)
              for i in range(1, len(self.layers) - 2) ])

        self.linears.apply(self.param_initialization)
            
    #  Initialization for parameters
    def param_initialization(self, m):
        torch.manual_seed(self.retrain) # fix the seed
        
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
                # kaiming is suggested only for relu or leaky_relu
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
        x = activation(self.linears[0](x), self.activation_str)
        
        for i in range(1, len(self.layers) - 2):
            x = activation(
                    self.linears[i](
                        self.batch_norm[i-1](x) ), self.activation_str)
        
        return self.linears[-1](x)
    
#########################################
# FNN_LN class
#########################################         
class FNN_LN(nn.Module):

    def __init__(self, layers, activation_str, initialization_str):
        super().__init__()
        self.layers = layers # list with the number of neurons for each layer
        self.activation_str = activation_str
        self.retrain = retrain
        self.initialization_str = initialization_str
        
        # fix the seed for retrain
        torch.manual_seed(self.retrain)
        
        # linear layers
        self.linears = nn.ModuleList(
            [ nn.Linear(self.layers[i], self.layers[i+1]) 
              for i in range( len(self.layers) - 1 ) ])
        
        # batch normalization apllied in hidden layers
        self.layer_norm = nn.ModuleList(
            [ nn.LayerNorm(self.layers[i], device = mydevice)
              for i in range(1, len(self.layers) - 2) ])

        self.linears.apply(self.param_initialization)
            
    #  Initialization for parameters
    def param_initialization(self, m):
        torch.manual_seed(self.retrain) # fix the seed
        
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
        x = activation(self.linears[0](x), self.activation_str)
        
        for i in range(1, len(self.layers) - 2):
            x = activation(
                    self.linears[i](
                        self.layer_norm[i-1](x) ), self.activation_str)
        
        return self.linears[-1](x)

#########################################
# DeepONet class
#########################################  
class DeepONet(nn.Module):
    def __init__(self, layers_dict, activation_dict):
        """ parameters are dictionaries """
        super().__init__()
        self.layers_b = layers_dict["branch"]
        self.layers_t = layers_dict["trunk"]
        self.act_b = activation_dict["branch"]
        self.act_t = activation_dict["trunk"]
        self.init_b = init_b
        self.init_t = init_t
        
        if arc_b == "FNN":
            self.branch  = FNN(self.layers_b, self.act_b)
        elif arc_b == "FNN_BN":
            self.branch  = FNN_BN(self.layers_b, self.act_b, self.init_b)
        elif arc_b == "FNN_LN":
            self.branch  = FNN_LN(self.layers_b, self.act_b, self.init_b)
            
        if arc_t == "FNN":
            self.trunk   = FNN(self.layers_t, self.act_t)
        elif arc_t == "FNN_BN":
            self.trunk  = FNN_BN(self.layers_t, self.act_t, self.init_t)
        elif arc_t == "FNN_LN":
            self.trunk  = FNN_LN(self.layers_t, self.act_t, self.init_t)
            
        # Final bias
        self.b = torch.nn.Parameter(torch.tensor(0.), requires_grad = True)

    def forward(self, b_in, t_in):
        # branch network
        b_in = self.branch(b_in)
        # trunk network
        t_in = activation(self.trunk(t_in), self.act_t)
        # sclar product
        out = torch.einsum("ij,kj->ik",b_in,t_in)
        # add final bias
        out += self.b     
        return out

if __name__ == '__main__':
    # to save the data
    writer = SummaryWriter(log_dir = name_log_dir )
    # available device
    print('Available device:', mydevice)
    
    #########################################
    # Preprocessing of the data
    #########################################    
    #### Index set
    g = torch.Generator().manual_seed(1) # fix the seed
    idx_tot = torch.randperm(ntot, device = 'cpu', generator = g)
    # expected cpu device for generator
    idx_train = idx_tot[:ntrain]
    idx_test = idx_tot[ntrain:ntrain + ntest]
    
    #### Training data
    tspan_train, a_train, u_train = MatReader(DataPath)    
    tspan_train = tspan_train[::s].unsqueeze(-1).to(mydevice)
    a_train, u_train = a_train[idx_train, ::s], u_train[idx_train, ::s]
    
    #### Test data
    tspan_test, a_test, u_test = MatReader(DataPath)
    tspan_test = tspan_test[::s].unsqueeze(-1).to(mydevice)
    a_test, u_test = a_test[idx_test, ::s], u_test[idx_test, ::s]
    
    #### normalize data 
    if data_norm:
        a_normalizer = UnitGaussianNormalizer(a_train)
        a_train_pp = a_normalizer.encode(a_train) # pp = PostProcessed
        a_test_pp = a_normalizer.encode(a_test)
    
    #### batch loader
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a_train_pp, u_train),
                                                batch_size = batch_size)
    
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a_test_pp, u_test),
                                              batch_size = batch_size)    
    print('Data loaded')
    
    ################################################################
    # training, evaluation and plot
    ################################################################
    #### DeepONet parameters
    layers = {"branch" : [u_dim] + inner_layer_b + [G_dim],
              "trunk"  : [x_dim + N_FourierF] + inner_layer_t + [G_dim] }
    activ  = {"branch" : activation_b,
              "trunk"  : activation_t}
    
    # Inizialize the model
    model = DeepONet(layers, activ)
    model.to(mydevice)
    
    # Count the parameters
    par_tot = sum(p.numel() for p in model.parameters())
    print("Total DeepONet parameters: ", par_tot)
    writer.add_text("Parameters", 'Total parameters number: ' + str(par_tot), 0)

    # AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    
    # lr policy
    if scheduler == "StepLR":
        # halved the learning rate every 100 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
    if scheduler == "CosineAnnealingLR":
        # Cosine Annealing Scheduler (SGD with warm restart)
        iterations = epochs*(ntrain//batch_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = iterations)
    
    # Loss function
    if Loss == 'L2':
        myloss = L2relLoss()
    elif Loss == 'H1':
        myloss = H1relLoss()
    elif Loss == 'mse':
        myloss = MSE()
    
    t1 = default_timer()
    # Training process
    for ep in range(epochs+1):
        #### One epoch of training
        model.train()
        train_loss = 0
        for step, (a, u) in enumerate(train_loader):
            a, u = a.to(mydevice), u.to(mydevice)
            
            optimizer.zero_grad() # annealing the gradient
            
            out = model.forward(a, tspan_train) # compute the outpu
            
            # compute the loss
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
        test_mse = 0.0
        test_h1 = 0.0
        with torch.no_grad():
            for a, u in test_loader:
                a, u = a.to(mydevice), u.to(mydevice)
    
                out = model.forward(a, tspan_test)  
                
                test_l2 += L2relLoss()(out, u).item()
                test_mse += MSE()(out, u).item()
                test_h1 += H1relLoss()(out, u).item()
                
        train_loss /= ntrain
        test_l2 /= ntest
        test_mse /= ntest
        test_h1 /= ntest
        
        t2 = default_timer()
        if ep % 100 == 0:
            print('Epoch:', ep, 
                  'Time:', t2-t1,
                  'Train_loss_'+Loss+':', train_loss, 
                  'Test_rel_loss_l2:', test_l2,
                  'Test_mse:', test_mse, 
                  'Test_rel_loss_h1:', test_h1
                  )
            
        writer.add_scalars('FNO_HH', {'Train_loss': train_loss,
                                                'Test_rel_loss_l2': test_l2,
                                                'Test_mse': test_mse,
                                                'Test_rel_loss_h1': test_h1}, ep)
    
        #########################################
        # plot data at every epoch_step
        #########################################
        if ep == 0:
            #### initial value of a
            esempio_test = a_test[idx, :].to('cpu')
            
            X = np.linspace(0, 100, esempio_test.shape[1])
            fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
            fig.suptitle('Applied current (I_app)')
            ax[0].set(ylabel = 'I_app(t)')
            for i in range(n_idx):
                # ax[i].plot(X, scale_data(esempio_test[i], 0, 1, a_data_min, a_data_max))
                ax[i].plot(X, esempio_test[i])
                ax[i].set(xlabel = 't')
                ax[i].set_ylim([-0.2, 10])
                ax[i].grid()
            if plotting:
                plt.show()
            writer.add_figure('Applied current (I_app)', fig, 0)

            #### Approximate classical solution
            soluzione_test = u_test[idx]
            # soluzione_test = scale_data(soluzione_test, 0, 1, u_data_min, u_data_max).to('cpu')
            X = np.linspace(0, 100, soluzione_test.shape[1])
            fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
            fig.suptitle('Numerical approximation (V_m)')
            ax[0].set(ylabel = 'V_m (mV)')
            for i in range(n_idx):
                ax[i].plot(X, soluzione_test[i])
                ax[i].set(xlabel = 't')
                ax[i].grid()
            if plotting:
                plt.show()
            writer.add_figure('Numerical approximation (V_m)', fig, 0)

        #### approximate solution with FNO of HH model
        if ep % ep_step == 0:
            with torch.no_grad():  # no grad for effeciency reason
                out_test = model(esempio_test.to(mydevice), tspan_test)
                # out_test = scale_data(out_test, 0, 1, u_data_min, u_data_max)
                out_test = out_test.to('cpu')

            # X = np.linspace(0, 1, out_test.shape[1])
            fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
            fig.suptitle('FNO approximation (V_m)')
            ax[0].set(ylabel = 'V_m (mV)')
            for i in range(n_idx):
                ax[i].plot(X, out_test[i])
                ax[i].set(xlabel = 't')
                ax[i].grid()
            if plotting:
                plt.show()
            writer.add_figure('FNO approximation (V_m)', fig, ep)

            #### Module of the difference between classical anf FNO approximation
            diff = torch.abs(out_test - soluzione_test)
            # X = np.linspace(0, 1, diff.shape[1])
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
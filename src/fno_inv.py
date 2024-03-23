"""
fno_inv.py

author: Edoardo Centofanti
credits to : Yahya Saleh and Massimiliano Ghiotto

classes relative to adaptive-FNO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .architectures import MLP, activation

#########################################
# fourier layer
#########################################

class LinearLayer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        
        # more parameters: parameters for each one of the 32 vectors
        #self.weights = nn.Parameter(torch.ones(shape[0], shape[1]),requires_grad=True)
        #self.bias = nn.Parameter(torch.zeros(shape[0], shape[1]),requires_grad=True)
        
        # parameters shared between all the 32 vectors of size 510
        self.weights = nn.Parameter(torch.zeros(shape[1], dtype=torch.float),requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(shape[1], dtype=torch.float),requires_grad=True)
        
        self.factor = 1e-2
    def forward(self, x, mode="direct"):
        a = self.weights**2+1

        if mode == "direct":
            #x = a * x + self.bias 
            #x = torch.einsum('jk,ijk->ijk',a,x)
            x = torch.einsum('k,ijk->ijk',a,x)
            x = x + self.bias
            #print(x.shape)
        elif mode == "inverse":
            x = x - self.bias
            #x = torch.einsum('ijk,jk->ijk',x,1./a)
            x = torch.einsum('ijk,k->ijk',x,1./a)
            #print(x.shape)  
        else:
            ValueError('Invalid mode! It can be either direct or inverse')
        return x    

    def derivative(self, x):
        return self.weights**2+1
    
class AdFourierLayer(nn.Module):
    def __init__(self, datasize, in_channels, out_channels, modes, weights_norm, scalar, act_fun):
        super().__init__()
        """
        1D Adaptive Fourier layer. We initialize the parameter for the Fourier layer  
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.weights_norm = weights_norm
        self.scalar = scalar
        self.activation = act_fun
        self.g = LinearLayer((in_channels,datasize)) # shape (32,510)
        
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
                    mode = 'fan_in', nonlinearity = act_fun.lower())
               
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
                        dtype = torch.float)), mode = 'fan_in', nonlinearity = act_fun.lower())

    
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
        # perform adaptive strategy
        x = self.g(x,mode="inverse") / self.g.derivative(x)
        # Compute Fourier coefficients up to factor of e^(- something constant)
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
# FNO for 1d model
#########################################
class AdFNO1d(nn.Module):
    """ 
    Fourier Neural Operator for approximate the Hodgkin-Huxley model
    """
    def __init__(self, datasize, d_a, d_v, d_u, L, modes, act_fun, 
                 initialization, scalar, padding,
                 arc, x_padding, RNN):
        """ 
        
        d_a : int
            input dimension
            
        d_v : int
            hidden dimension        
            
        d_u : int
            equal to output shape dimension 
            
        L: int
           depth of FNO
          
        modes : int
            equal to k_{max, 1}
        
        act_fun: string
            act_fun function
        
        padding: bool
            True if you want padding, False otherwise
            
        """
        super().__init__()      
        
        self.d_a = d_a
        self.d_v = d_v
        self.d_u = d_u
        self.L = L
        self.arc = arc
        self.modes = modes
        self.activation = activation(act_fun)
        self.padding = padding
        if self.padding:
            self.x_padding = x_padding
        self.RNN = RNN
        
        #### Lifting operator
        self.p = nn.Linear(d_a, d_v) # input features is d_a = 2: (a(t), t)
        
        #### Fourier operator
        if self.RNN:
            self.fouriers = AdFourierLayer(datasize, d_v, d_v, modes, initialization, scalar, act_fun) 
            
            self.ws = nn.Linear(d_v, d_v)
        else:
            self.fouriers = nn.ModuleList([
                AdFourierLayer(datasize, d_v, d_v, modes, initialization, scalar, act_fun) 
                for _ in range(L) ])
            
            self.ws = nn.ModuleList([ nn.Linear(d_v, d_v) for _ in range(L) ])
        
        # classic architecture
        if arc == "Classic":           
            #### Projection operator
            self.q = nn.Linear(d_v, d_u) # output features is d_u: u(x,y)
            
        elif arc == "Zongyi":
            #### Fourier operator
            if self.RNN:
                self.mlps = MLP(d_v, d_v, d_v)
            else:
                self.mlps = nn.ModuleList([ MLP(d_v, d_v, d_v) for _ in range(L) ])
            
            #### Projection operator
            self.q = MLP(d_v, d_u, 4*d_u)

    def forward(self, x):
        # preprocessing
        if self.d_a==2: 
            v, x = x[0], x[1]
            v = v.unsqueeze(-1)
            x = x.reshape(1,x.size(0),1).repeat([v.size(0), 1, 1])
            x = torch.cat((v,x),dim=-1)
        elif self.d_a==3:
            v, x = x[0], x[1]
            v = v.permute(0, 2, 1)
            x = x.reshape(1,x.size(0),1).repeat([v.size(0), 1, 1])
            x = torch.cat((v,x),dim=-1)
        else:
            raise ValueError("d_a dimension invalid.") 

        # initially x.size() = (n_samples)*(n_t)*(d_a)
        
        #### Apply lifting operator P
        x = self.p(x)          # x.size() = (n_samples)*(n_t)*(d_v)
        x = x.permute(0, 2, 1) # x.size() = (n_samples)*(d_v)*(n_t)
        
        if self.padding:
            x = F.pad(x, [0, self.x_padding])
        
        #### Fourier Layers
        # classic architecture
        if self.arc == "Classic":
            for i in range(self.L):
                if self.RNN:
                    x1 = self.fouriers(x)
                    x2 = self.ws(x.permute(0, 2, 1))
                    x = x1 + x2.permute(0, 2, 1)
                    if i < self.L - 1:
                        x = self.activation(x)
                else:
                    x1 = self.fouriers[i](x)
                    x2 = self.ws[i](x.permute(0, 2, 1))
                    x = x1 + x2.permute(0, 2, 1)
                    if i < self.L - 1:
                        x = self.activation(x)
        # Zongyi Li architecture
        elif self.arc == "Zongyi":
            for i in range(self.L):
                if self.RNN:
                    x1 = self.fouriers(x)
                    x1 = self.mlps(x1.permute(0, 2, 1))
                    x2 = self.ws(x.permute(0, 2, 1))
                    x = x1 + x2
                    x = x.permute(0, 2, 1)
                    if i < self.L - 1:
                        x = self.activation(x)   
                else:
                    x1 = self.fouriers[i](x)
                    x1 = self.mlps[i](x1.permute(0, 2, 1))
                    x2 = self.ws[i](x.permute(0, 2, 1))
                    x = x1 + x2
                    x = x.permute(0, 2, 1)
                    if i < self.L - 1:
                        x = self.activation(x)           
                    
        if self.padding:
            x = x[..., :-self.x_padding]
            
        #### apply projection operator Q
        x = self.q(x.permute(0, 2, 1))
        return x.squeeze()
    
if __name__=="__main__":
    ax    = torch.rand(200,500)
    x     = torch.rand(500,1)
    
    d_a            = 2 
    d_v            = 32
    d_u            = 1
    L              = 4
    modes          = 16
    act_fun        = "ReLu"
    initialization = "Kaiming"
    scalar         = "Real" 
    padding        = True
    arc            = "Classic"
    x_padding      = 10
    RNN            = True 

    model = AdFNO1d(d_a,d_v,d_u,L,modes,act_fun,initialization,scalar,padding,arc,x_padding,RNN)
    out = model.forward((ax,x))
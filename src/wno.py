"""
This code is extracted from the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet neural operator: a neural 
   operator for parametric partial differential equations. arXiv preprint arXiv:2205.02191.

modified by: Edoardo Centofanti
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1D, IDWT1D

torch.manual_seed(0)
np.random.seed(0)

""" Def: 1d Wavelet layer """
class WaveConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, level, dummy, wavelet):
        super(WaveConv1d, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.wavelet = wavelet  #'db20'
        self.dwt_ = DWT1D(wave=self.wavelet, J=self.level, mode='symmetric').to(dummy.device)
        self.mode_data, self.mode_coeff = self.dwt_(dummy)
        self.modes1 = self.mode_data.shape[-1]
        self.modes2 = self.mode_coeff[-2].shape[-1]
        self.modes3 = self.mode_coeff[-3].shape[-1]
        
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes3))

    # Convolution
    def mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute single tree Discrete Wavelet coefficients using some wavelet     
        dwt = DWT1D(wave=self.wavelet, J=self.level, mode='symmetric').to(x.device)
        x_ft, x_coeff = dwt(x)
        
        # Multiply the final low pass and high pass coefficients
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1],  device=x.device)
        out_ft = self.mul1d(x_ft, self.weights1)
        x_coeff[-1] = self.mul1d(x_coeff[-1].clone(), self.weights2)
        x_coeff[-2] = self.mul1d(x_coeff[-2].clone(), self.weights3)
        x_coeff[-3] = self.mul1d(x_coeff[-3].clone(), self.weights4)
        
        # Reconstruct the signal
        idwt = IDWT1D(wave=self.wavelet, mode='symmetric').to(x.device)
        x = idwt((out_ft, x_coeff))        
        return x
        
""" The forward operation """
class WNO1d(nn.Module):
    def __init__(self, width, level, dummy_data):
        super(WNO1d, self).__init__()

        """
        The WNO network. It contains 4 layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. 4 layers of the integral operators v(+1) = g(K(.) + W)(v).
            W is defined by self.w_; K is defined by self.conv_.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.level = level
        self.width = width
        self.dummy_data = dummy_data
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(dummy_data.shape[1], self.width)

        self.conv0 = WaveConv1d(self.width, self.width, self.level, self.dummy_data, 'db24')
        self.conv1 = WaveConv1d(self.width, self.width, self.level, self.dummy_data, 'db24')
        self.conv2 = WaveConv1d(self.width, self.width, self.level, self.dummy_data, 'db24')
        self.conv3 = WaveConv1d(self.width, self.width, self.level, self.dummy_data, 'db24')
        self.conv4 = WaveConv1d(self.width, self.width, self.level, self.dummy_data, 'db24')
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        # preprocessing 
        v, x = x[0], x[1]
        if len(v.shape)==2:
            v = v.unsqueeze(-1)
        else:
            v = v.permute(0, 2, 1)
        x = x.reshape(1,x.size(0),1).repeat([v.size(0), 1, 1])
        x = torch.cat((v,x),dim=-1)  

        # proper forward pass
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # do padding, if required

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv4(x)
        x2 = self.w4(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # remove padding, when required
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.squeeze(-1)
        return x
    
if __name__=="__main__":
    ax    = torch.rand(200,500)
    x     = torch.rand(500,1)
    dummy = torch.rand(1,2,x.size(0))

    level = 6
    width = 48
    
    model = WNO1d(width,level,dummy)
    out = model.forward((ax,x))
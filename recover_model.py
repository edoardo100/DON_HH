# -*- coding: utf-8 -*-
"""
deepONet_HH_pytorch.py

author: Edoardo Centofanti

Learning Hodgkin-Huxley model with DeepONet
"""
# internal modules
from src.utility_dataset import *
from src.architectures import get_optimizer, get_loss
from src.don import DeepONet
from src.wno import WNO1d
from src.fno import FNO1d
from src.training import Training
# external modules
import torch
# for test launcher interface
import os
import yaml
import argparse
import matplotlib.pyplot as plt
# tensorboard and plotting
from tensorboardX import SummaryWriter

#########################################
# default value
#########################################
mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#mydevice = 'cpu'
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

# Define command-line arguments
parser = argparse.ArgumentParser(description="Learning Hodgkin-Huxley model with DeepONet")
parser.add_argument("--config_file", type=str, default="default_params_don.yml", help="Path to the YAML configuration file")
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
arc           = config.get("arc")
dataset_train = config.get("dataset_train")
dataset_test  = config.get("dataset_test")
batch_size    = config.get("batch_size")
scaling       = config.get("scaling")
labels        = config.get("labels")      # default False
full_v_data   = config.get("full_v_data") # default False
N_FourierF    = config.get("N_FourierF")
scale_FF      = 1  # config.get("scale_FF")
adapt_actfun  = config.get("adapt_actfun")
scheduler     = config.get("scheduler")
Loss          = config.get("Loss")
epochs        = config.get("epochs")
lr            = config.get("lr")
u_dim         = config.get("u_dim")
x_dim         = config.get("x_dim")
G_dim         = config.get("G_dim")
inner_layer_b = config.get("inner_layer_b")
inner_layer_t = config.get("inner_layer_t")
activation_b  = config.get("activation_b")
activation_t  = config.get("activation_t")
arc_b         = config.get("arc_b")
arc_t         = config.get("arc_t") 
initial_b     = config.get("initial_b")
initial_t     = config.get("initial_t")
#### WNO parameters
width = config.get("width")
level = config.get("level")
#### FNO parameters
d_a = config.get("d_a")
d_v = config.get("d_v")
d_u = config.get("d_u")
L = config.get("L")
modes = config.get("modes")
act_fun = config.get("act_fun")
initialization = config.get("initialization")
scalar = config.get("scalar")
padding = config.get("padding")
arc_fno = config.get("arc_fno")
x_padding = config.get("x_padding")
RNN = config.get("RNN")
#### Plotting parameters
show_every = config.get("show_every")
ep_step    = config.get("ep_step")
idx        = config.get("idx")
plotting   = config.get("plotting")

#########################################
#                 MAIN
#########################################
if __name__=="__main__":
    # [159, 69, 134, 309]
    idx = torch.randint(low=0, high=400, size=(4,))
    # 125
    idx = torch.tensor([159, 69, 134, 309])
    print("indexes to print = "+str(idx))
    # Load dataset
    if "LR" in dataset_train:
        u_train, x_train, v_train, scale_fac = load_LR_train(dataset_train,full_v_data)
        u_test, x_test, v_test, indices = load_LR_test(dataset_test,full_v_data)
    else:
        u_train, x_train, v_train, scale_fac, _ = load_train(dataset_train,scaling,labels,full_v_data,shuffle=True)
        u_test, x_test, v_test, indices = load_test(dataset_test,scale_fac,scaling,labels,full_v_data,shuffle=True)
        
    # batch loader
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(v_train, u_train),
                                                batch_size = batch_size)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(v_test, u_test),
                                              batch_size = batch_size) 
    
    # Loss function
    myloss = get_loss(Loss)

    modelname = "peano_test/model_" + param_file_name.replace("peano_test/", "")

    model = torch.load(modelname, map_location=torch.device('cpu'))

    #### initial value of v
    u_test_unscaled, x_test_unscaled, v_test_unscaled = load_test(dataset_test,full_v_data=True)
    # Same order of scaled data
    u_test_unscaled = u_test_unscaled[indices]
    v_test_unscaled = v_test_unscaled[indices]

    esempio_test    = v_test_unscaled[idx, :].to('cpu')
    esempio_test_pp = v_test[idx, :].to('cpu')
    sol_test        = u_test_unscaled[idx]
    x_test_unscaled = x_test_unscaled.to('cpu')
    
    ## Third figure for approximation with DON of HH model
    with torch.no_grad():  # no grad for efficiency reasons
        out_test = model((esempio_test_pp, x_test))
        out_test = out_test.to('cpu')

    # Create a single figure with a grid layout
    fig, axs = plt.subplots(2, len(idx), figsize=(18, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # First row: Numerical approximation (V_m) and DON approximation (V_m)
    for i in range(len(idx)):
        axs[0, i].plot(x_test_unscaled, sol_test[i].to('cpu'), label='Numerical approximation')
        axs[0, i].plot(x_test_unscaled, out_test[i], 'r--', label=arc+' approximation')
        axs[0, 0].set_ylabel('$V_m$ (mV)', labelpad=-5)
        axs[0, i].set_xlabel('t')
        axs[0, i].set_ylim([-100, 35])
        axs[0, i].grid()
        axs[0, i].legend()
    
    # Second row: Applied current (I_app)
    for i in range(len(idx)):
        axs[1, i].plot(x_test_unscaled, esempio_test[i])
        axs[1, 0].set_ylabel('$I_{app}$(t)')
        axs[1, i].set_xlabel('t')
        axs[1, i].set_ylim([-0.2, 10])
        axs[1, i].grid()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
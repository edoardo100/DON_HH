# -*- coding: utf-8 -*-
"""
deepONet_HH_pytorch.py

author: Edoardo Centofanti

Learning Hodgkin-Huxley model with DeepONet
"""
# internal modules
from src.utility_dataset import *
from src.architectures import get_optimizer, get_loss
from src.don import DeepONet, DeepONet_md
from src.fno_inv import AdFNO1d
from src.wno import WNO1d
from src.fno import FNO1d
from src.training import Training
# external modules
import torch
# for test launcher interface
import os
import yaml
import argparse

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
# Hyperparameter
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
# DeepONet hyper-parameters
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
#### WNO hyper-parameters
width = config.get("width")
level = config.get("level")
#### FNO hyper-parameters
d_a            = config.get("d_a")
d_v            = config.get("d_v")
d_u            = config.get("d_u")
L              = config.get("L")
modes          = config.get("modes")
act_fun        = config.get("act_fun")
initialization = config.get("initialization")
scalar         = config.get("scalar")
padding        = config.get("padding")
arc_fno        = config.get("arc_fno")
x_padding      = config.get("x_padding")
RNN            = config.get("RNN")
#### Plotting hyper-parameters
show_every = config.get("show_every")
ep_step    = config.get("ep_step")
idx        = config.get("idx")
plotting   = config.get("plotting")

#########################################
#                 MAIN
#########################################
if __name__=="__main__":
    
    #### Network parameters
    layers, activ, init = None, None, None
    
    if arc=="DON" or arc=="DON_md": # For DON or DON multidimensional
        layers = {"branch" : [u_dim] + inner_layer_b + [G_dim],
                  "trunk"  : [x_dim*(N_FourierF==0) + 2*N_FourierF] + inner_layer_t + [G_dim] }
        activ  = {"branch" : activation_b,
                  "trunk"  : activation_t}
        init   = {"branch" : initial_b,
                  "trunk"  : initial_t}
    
    # Load dataset
    if "LR" in dataset_train:
        u_train, x_train, v_train, scale_fac = load_LR_train(dataset_train,full_v_data)
        u_test, x_test, v_test, indices = load_LR_test(dataset_test,full_v_data)
    elif arc=="DON_md" or arc=="FNO_md": # for the moment we fix the dataset names because this feature is in beta version 
        names_train = [
            'dataset/datasetHH_test2_train.mat',
            'dataset/datasetHH_test2_train_h.mat',
            'dataset/datasetHH_test2_train_m.mat',
            'dataset/datasetHH_test2_train_n.mat'
        ]
        names_test = [
            'dataset/datasetHH_test2_test.mat',
            'dataset/datasetHH_test2_test_h.mat',
            'dataset/datasetHH_test2_test_m.mat',
            'dataset/datasetHH_test2_test_n.mat'
        ]
        u_train_v, x_train, v_train, scale_fac, _ = load_train(names_train[0],scaling,labels,full_v_data,shuffle=True)
        u_test_v, x_test, v_test, indices = load_test(names_test[0],scale_fac,scaling,labels,full_v_data,shuffle=True)
        u_train_h, _, _, _, _ = load_train(names_train[1],scaling,labels,full_v_data,shuffle=True)
        u_test_h, _, _, _ = load_test(names_test[1],scale_fac,scaling,labels,full_v_data,shuffle=True)
        u_train_m, _, _, _, _ = load_train(names_train[2],scaling,labels,full_v_data,shuffle=True)
        u_test_m, _, _, _ = load_test(names_test[2],scale_fac,scaling,labels,full_v_data,shuffle=True)
        u_train_n, _, _, _, _ = load_train(names_train[3],scaling,labels,full_v_data,shuffle=True)
        u_test_n, _, _, _ = load_test(names_test[3],scale_fac,scaling,labels,full_v_data,shuffle=True)
        u_train = torch.stack((u_train_v,u_train_h,u_train_m,u_train_n),dim=0) # shape(4,1600,500)
        u_test  = torch.stack((u_test_v,u_test_h,u_test_m,u_test_n),dim=0)
        u_train = u_train.permute(1,2,0) # shape(1600,500,4). Necessary for DataLoader
        u_test  = u_test.permute(1,2,0)
    else:
        u_train, x_train, v_train, scale_fac, _ = load_train(dataset_train,scaling,labels,full_v_data,shuffle=True)
        u_test, x_test, v_test, indices = load_test(dataset_test,scale_fac,scaling,labels,full_v_data,shuffle=True)
        
    # batch loader
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(v_train, u_train),
                                                batch_size = batch_size, shuffle=True, generator = torch.Generator(device=mydevice))
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(v_test, u_test),
                                              batch_size = batch_size) 
    
    model = None
    if arc=="DON":
        model = DeepONet(layers,activ,init,arc_b,arc_t,adapt_actfun)
    elif arc=="DON_md":
       dim   = 4 # we have 4 equations
       model = DeepONet_md(dim,layers,activ,init,arc_b,arc_t,adapt_actfun) 
    elif arc=="WNO":
        if not full_v_data:
            raise ValueError("full_v_data must be true")
        if "LR" in dataset_train:
            dummy = torch.rand(1,3,x_train.size(0))
        else:
            dummy = torch.rand(1,2,x_train.size(0))
        model = WNO1d(width,level,dummy)
    elif arc=="FNO" or arc=="FNO_md":
        if not full_v_data:
            raise ValueError("full_v_data must be true")
        model = FNO1d(d_a,d_v,d_u,L,modes,act_fun,initialization,scalar,padding,arc_fno,x_padding,RNN)
    elif arc=="AdFNO" or arc=="AdFNO_md":
        if full_v_data==False:
            raise ValueError("full_v_data must be true")
        datasize = v_test.shape[1] + x_padding
        model = AdFNO1d(datasize,d_a,d_v,d_u,L,modes,act_fun,initialization,scalar,padding,arc_fno,x_padding,RNN)
    else:
        raise ValueError("This architecture has not been implemented.")
    # Count the parameters
    par_tot = sum(p.numel() for p in model.parameters())
    print("Total trainable parameters: ", par_tot)

    optimizer, schedulerName, scheduler = get_optimizer(model,lr,scheduler,epochs,u_train.shape[0],batch_size)

    # Loss function
    myloss = get_loss(Loss)

    trainer = Training(
        model,
        epochs,
        optimizer,
        schedulerName,
        scheduler,
        myloss,
        dataset_test,
        ntrain=u_train.shape[0],
        ntest=u_test.shape[0],
        indices=indices,
        train_loader=train_loader,
        test_loader=test_loader,
        x_train=x_train,
        x_test=x_test,
	    device=mydevice,
        show_every=show_every
    )

    trainer.train()

    torch.save(model, name_model)

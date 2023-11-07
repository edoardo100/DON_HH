# -*- coding: utf-8 -*-
"""
deepONet_HH_pytorch.py

author: Edoardo Centofanti

Learning Hodgkin-Huxley model with DeepONet
"""
# internal modules
from src.utility_dataset import *
from src.architectures import L2relLoss, MSE, DeepONet
# external modules
import torch
from timeit import default_timer
import scipy.io as sio
# for test launcher interface
import os
import yaml
import argparse
# tensorboard and plotting
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

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

# Define command-line arguments
parser = argparse.ArgumentParser(description="Learning Hodgkin-Huxley model with DeepONet")
parser.add_argument("--config_file", type=str, default="beta_test_006.yml", help="Path to the YAML configuration file")
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
dataset_train = config["dataset_train"]
dataset_test  = config["dataset_test"]
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
arc_b         = config["arc_b"] 
arc_t         = config["arc_t"] 
initial_b     = config["initial_b"]
initial_t     = config["initial_t"]
#### Plotting parameters
ep_step  = config["ep_step"]
idx      = config["idx"]
n_idx    = len(idx)
plotting = config["plotting"]

#########################################
#                 MAIN
#########################################
if __name__=="__main__":

    dataname = dataset_train 
    dataname_test = dataset_test 
    
    labels      = False
    full_v_data = False

    writer = SummaryWriter(log_dir = name_log_dir )
    
    #### Network parameters
    layers = {"branch" : [u_dim] + inner_layer_b + [G_dim],
              "trunk"  : [x_dim] + inner_layer_t + [G_dim] }
    activ  = {"branch" : activation_b,
              "trunk"  : activation_t}
    init   = {"branch" : initial_b,
              "trunk"  : initial_t}
    
    # Load dataset
    u_train, x_train, v_train, scale_fac, _ = load_train(dataname,scaling,labels,full_v_data,shuffle=True)
    u_test, x_test, v_test, indices = load_test(dataname_test,scale_fac,scaling,labels,full_v_data,shuffle=True)

    # batch loader
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(v_train, u_train),
                                                batch_size = batch_size)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(v_test, u_test),
                                              batch_size = batch_size) 
    
    model = DeepONet(layers,activ,init,arc_b,arc_t)

    # Count the parameters
    par_tot = sum(p.numel() for p in model.parameters())
    print("Total DeepONet parameters: ", par_tot)
    writer.add_text("Parameters", 'Total parameters number: ' + str(par_tot), 0)

    # AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-4)
    # lr policy
    if scheduler == "StepLR":
        # halved the learning rate every 100 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10000, gamma = 0.5)
    if scheduler == "CosineAnnealingLR":
        # Cosine Annealing Scheduler (SGD with warm restart)
        iterations = epochs*(u_train.shape[0]//batch_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = iterations)

    # Loss function
    if Loss == "L2":
        myloss = L2relLoss()
    elif Loss == "mse":
        myloss = MSE()
    #elif Loss == 'H1':
    #    myloss = H1relLoss()
    t1 = default_timer()
    # Unscaled dataset (for plotting)
    u_test_unscaled, x_unscaled, v_test_unscaled = load_test(dataname_test)
    # Same order of scaled data
    u_test_unscaled = u_test_unscaled[indices]
    v_test_unscaled = v_test_unscaled[indices]
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
        test_l2  = 0.0
        test_mse = 0.0
        #test_h1 = 0.0
        with torch.no_grad():
            for v, u in test_loader:
                v, u = v.to(mydevice), u.to(mydevice)
    
                out = model.forward((v,x_test))      
                
                # rescaling to compute the test error
                if scaling == "Default":
                    out = unscale_data(out.to(mydevice),scale_fac[0],scale_fac[1])
                    u = unscale_data(u.to(mydevice),scale_fac[0],scale_fac[1])
                elif scaling == "Gaussian":
                    out = inverse_gaussian_scale(out.to(mydevice),scale_fac[0],scale_fac[1])
                    u = inverse_gaussian_scale(u.to(mydevice),scale_fac[0],scale_fac[1])
                elif scaling == "Mixed":
                    out = inverse_gaussian_scale(out.to(mydevice),scale_fac[0],scale_fac[1])
                    u = inverse_gaussian_scale(u.to(mydevice),scale_fac[0],scale_fac[1])
                    
                test_l2 += L2relLoss()(out, u).item()
                test_mse += MSE()(out, u).item()
                #test_h1 += H1relLoss()(out, u).item()
                
        train_loss /= u_train.shape[0]
        test_l2 /= u_test.shape[0]
        test_mse /= u_test.shape[0]
        #test_h1 /= u_test.shape[0]
    
        t2 = default_timer()
        if ep%100==0:
            print('Epoch:', ep, 'Time:', t2-t1,
                  'Train_loss_'+Loss+':', train_loss, 'Test_loss_l2:', test_l2,
                  'Test_mse:', test_mse, 
                  #'Test_loss_h1:', test_h1
                  )

            writer.add_scalars('DON_HH', {'Train_loss': train_loss,
                                                    'Test_loss_l2': test_l2,
                                                    'Test_mse': test_mse
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
                ax[i].plot(X, soluzione_test[i].to('cpu'))
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
                out_test = unscale_data(out_test.to(mydevice),scale_fac[0],scale_fac[1])
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

    writer.flush() # to save final data
    writer.close() # close tensorboard writer
    torch.save(model, name_model)
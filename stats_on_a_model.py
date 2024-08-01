# -*- coding: utf-8 -*-
"""
deepONet_HH_pytorch.py

author: Edoardo Centofanti

Learning Hodgkin-Huxley model with DeepONet
"""
# internal modules
from src.utility_dataset import *
from src.architectures import get_optimizer, get_loss
from src.architectures import L2relLoss, L2relLossMultidim
# external modules
import torch
import statistics
# for test launcher interface
import os
import yaml
import argparse
import matplotlib.pyplot as plt

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
    # [159, 69, 258, 309]
    idx = torch.tensor([i for i in range(400)])
    # Load dataset
    # if "LR" in dataset_train:
    #     u_test, x_test, v_test, indices = load_LR_test(dataset_test,full_v_data)
    # else:
    #     u_test, x_test, v_test, indices = load_test(dataset_test,scale_fac,scaling,labels,full_v_data,shuffle=True)
        
    # Loss function
    myloss = get_loss(Loss)

    modelname = "peano_test/model_" + param_file_name.replace("peano_test/", "")

    model = torch.load(modelname, map_location=torch.device('cpu'))

    #### import dataset
    if arc=="DON_md" or arc=="FNO_md": # for the moment we fix the dataset names because this feature is in beta version 
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
        _, _, _, scale_fac, _ = load_train(names_test[0],scaling,labels,full_v_data,shuffle=True)
        _, x_test, v_test, indices = load_test(names_test[0],scale_fac,scaling,labels,full_v_data,shuffle=True)
        u_test_v, _, _ = load_test(names_test[0],full_v_data=True)
        u_test_h, _, _ = load_test(names_test[1],full_v_data=True)
        u_test_m, _, _ = load_test(names_test[2],full_v_data=True)
        u_test_n, _, _ = load_test(names_test[3],full_v_data=True)
        u_test_unscaled  = torch.stack((u_test_v,u_test_h,u_test_m,u_test_n),dim=0) # shape(4,1600,500)
        u_test_unscaled  = u_test_unscaled[:,indices,:]
        sol_test         = u_test_unscaled[:,idx,:].to('cpu')
    else:
        _, _, _, scale_fac, _ = load_train(dataset_train,scaling,labels,full_v_data,shuffle=True)
        _, x_test, v_test, indices = load_test(dataset_test,scale_fac,scaling,labels,full_v_data,shuffle=True)
        u_test_unscaled, _, _ = load_test(dataset_test,full_v_data=True)
        u_test_unscaled       = u_test_unscaled[indices]
        sol_test              = u_test_unscaled[idx].to('cpu')
    # Same order of scaled data
    esempio_test_pp = v_test[idx, :].to('cpu')
    
    with torch.no_grad():  # no grad for efficiency reasons
        out_test = model((esempio_test_pp, x_test))
        out_test = out_test.to('cpu')

    results = []
    for i in range(len(idx)):
        if "_md" in arc:
            err = L2relLossMultidim()(out_test[:,i,:].unsqueeze(1),sol_test[:,i,:].unsqueeze(1)).item()
        else:
            err = L2relLoss()(out_test[i].unsqueeze(0), sol_test[i].unsqueeze(0)).item()
        results.append((i,err))
        #print("Relative L2 for case "+str(i)+" = "+str(err))

    # Sort the results based on the error values
    results.sort(key=lambda x: x[1])
    threshold = 0.1 # remove, but count data > 15%

    # Extract the errors
    errors = [err for _, err in results]
    errors_thr = [i for i in errors if i < threshold]
    not_included = len(errors) - len(errors_thr)
    pcg_not_inc = not_included/len(errors)*100
    formatted_pcg = "{:.1f}".format(pcg_not_inc)
    median = statistics.median(errors)
    print("Median of the data for " + arc + ": " + str(median))
    # Change color of the bins depending on the architectures
    color = 'skyblue'
    if arc=='DON':
        color = 'skyblue'
    elif arc=='FNO':
        color = 'orange'
    elif arc=='WNO':
        color = 'green'
    # Create a histogram of the error values
    plt.figure(figsize=(6,4.5))
    label = formatted_pcg + "% test data have error > 10%"
    plt.hist(errors_thr, bins=50, edgecolor='white', color=color,label=label)
    plt.xlabel('Relative L2 Error', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Relative L2 Test Errors for ' + arc + '\nMedian = {:.3f}'.format(median), fontsize=16)
    plt.xticks(fontsize=12)  # Adjust fontsize for x ticks
    plt.yticks(fontsize=12)  # Adjust fontsize for y ticks

    # Print sorted results
    #print("Sorted Results (index, error):")
    listindex = [159, 69, 258, 309] # the same indexes shown in recover_model.py
    for index, err in results:
        if index in listindex:
            print(f"Index: {index}, Error: {err}")
    plt.legend(fontsize='large')
    plt.show()
"""
utility_dataset.py

author: Edoardo Centofanti

Functions for easier dataset manipulation
"""

import torch
import scipy.io as sio

"""
 Scaling data
"""

def scale_data(data, min_value, max_value):
    data_min = torch.min(data)
    data_max = torch.max(data)
    # Apply the linear transformation
    scaled_data = (max_value - min_value) * (data - data_min) / (data_max - data_min) + min_value
    return data_max, data_min, scaled_data

def scale_data_for_test(data, data_min, data_max, min_value, max_value):
    # Apply the linear transformation
    scaled_data = (max_value - min_value) * (data - data_min) / (data_max - data_min) + min_value
    return scaled_data

def unscale_data(scaled_data, original_max, original_min):
    # Map the unscaled data back to the original range
    # supposing that scaled_data is scaled in [0,1]
    unscaled_data = scaled_data * (original_max - original_min) + original_min
    return unscaled_data

def gaussian_scale(data, eps=1e-5):
    mean = torch.mean(data)
    std_dev = torch.std(data)
    scaled_data = (data - mean) / (std_dev + eps)
    return mean, std_dev, scaled_data

def gaussian_scale_for_test(data, mean, std_dev, eps=1e-5):
    scaled_data = (data - mean) / (std_dev + eps)
    return scaled_data

def inverse_gaussian_scale(scaled_data, original_mean, original_std_dev, eps=1e-5): 
    unscaled_data = scaled_data * (original_std_dev + eps) + original_mean
    return unscaled_data

"""
 Loading dataset: our dataset for the HH problem is composed by 14 different files:
 dataset_test_Npeaks.mat 
 with N=0..6 and 7 test dataset counterparts.
 Each file is composed by the following fields:
 - "iapps": applied currents, shape (1000,4) for train, (250,4) for test
   each row has [tin tf iapp n_peaks], with tin start time for pulse, 
   tf end time for pulse, iapp applied current intensity and n_peaks number of
   peaks in the solution for the potential
- "tspan": time span. In these cases it goes from 0 to 100 with 500 points. 
  Its shape is (1,500)
- "vs": solutions for the potential. Shape (1000,500) for train, (250,500) for test
"""

"""
Load single train: when you have a single file
"""
def load_single_train(dataname,scaling="None",labels=False,full_v_data=False):
    d         = sio.loadmat(dataname)
    u_data    = torch.tensor(d['vs']).float()
    x_data    = torch.tensor(d['tspan']).float()
    v_data1   = (torch.tensor(d['iapps'])[:,0:2]).float()   # pulse times
    v_data2   = (torch.tensor(d['iapps'])[:,[2]]).float()   # pulse intensities
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
    elif scaling == "Fourier": 
        # scaling used for FNO: x_data is not scaled
        u_mean, u_std = torch.mean(u_data), torch.std(u_data)
        x_min,  x_max = torch.min(x_data), torch.max(x_data)
        v_mean1, v_std1 = torch.mean(v_data1), torch.std(v_data1)
        # v_mean1, v_std1, v_data1 = gaussian_scale(v_data1) # pulse times are not scaled
        v_mean2, v_std2, v_data2 = gaussian_scale(v_data2)
        scale_fac = [u_mean,u_std,x_max,x_min,v_mean1,v_std1,v_mean2,v_std2]

    # Variant for v_data
    if full_v_data and labels:
        raise NotImplementedError("Full v and labels data is not implemented yet.")
    elif full_v_data:
        domain = x_data.flatten().repeat(v_data1.shape[0], x_data.shape[0])
        v_data = torch.where((domain >= v_data1[:, [0]]) & (domain <= v_data1[:, [1]]), 1.0, 0.0)
        v_data = v_data*v_data2
    elif labels:
        v_labels = (torch.tensor(d['iapps'])[:,[3]]).int()
        v_data   = torch.cat((v_data1,v_data2,v_labels),axis=1) 
    else:
        v_data = torch.cat((v_data1,v_data2),axis=1)

    return u_data, x_data.t(), v_data, scale_fac

"""
Load train: to merge different datasets
"""
def load_train(dataname, scaling=None, labels=False, full_v_data=False, shuffle=False):
    if isinstance(dataname,list):
        # Initialize empty lists to store the data
        all_u_data = []
        all_v_data = []
        scale_fac  = []
        x_data = None  # Initialize x_data

        for data in dataname:
            u_data, x_data, v_data, _ = load_single_train(data,labels=labels)
            # Append u_data and v_data to the lists
            all_u_data.append(u_data)
            all_v_data.append(v_data)

        # Stack the lists of tensors to form a single tensor
        all_u_data = torch.cat(all_u_data, dim=0)
        all_v_data = torch.cat(all_v_data, dim=0)
        v_data1   = all_v_data[:,0:2]   # pulse times
        v_data2   = all_v_data[:,[2]]   # pulse intensities

        if scaling == "Default":
            u_max, u_min, all_u_data = scale_data(all_u_data,0.0,1.0)
            x_max, x_min, x_data = scale_data(x_data,0.0,1.0)
            v_max1, v_min1, v_data1 = scale_data(v_data1,0.0,1.0)
            v_max2, v_min2, v_data2 = scale_data(v_data2,0.0,1.0)
            scale_fac = [u_max,u_min,x_max,x_min,v_max1,v_min1,v_max2,v_min2]
        elif scaling == "Gaussian":
            u_mean, u_std, all_u_data = gaussian_scale(all_u_data)
            x_mean, x_std, x_data = gaussian_scale(x_data)
            v_mean1, v_std1, v_data1 = gaussian_scale(v_data1)
            v_mean2, v_std2, v_data2 = gaussian_scale(v_data2)
            scale_fac = [u_mean,u_std,x_mean,x_std,v_mean1,v_std1,v_mean2,v_std2]
        elif scaling == "Mixed":
            u_mean, u_std, all_u_data = gaussian_scale(all_u_data)
            x_max, x_min, x_data = scale_data(x_data,0.0,1.0)
            v_mean1, v_std1, v_data1 = gaussian_scale(v_data1)
            v_mean2, v_std2, v_data2 = gaussian_scale(v_data2)
            scale_fac = [u_mean,u_std,x_max,x_min,v_mean1,v_std1,v_mean2,v_std2]
        elif scaling == "Fourier": 
            # scaling used for FNO: x_data is not scaled
            u_mean, u_std = torch.mean(u_data), torch.std(u_data)
            x_min,  x_max = torch.min(x_data), torch.max(x_data)
            v_mean1, v_std1 = torch.mean(v_data1), torch.std(v_data1)
            v_mean2, v_std2, v_data2 = gaussian_scale(v_data2)
            scale_fac = [u_mean,u_std,x_max,x_min,v_mean1,v_std1,v_mean2,v_std2]

        # Variant for v_data
        if full_v_data and labels:
            raise NotImplementedError("Full v and labels data is not implemented yet.")
        elif full_v_data:
            domain = x_data.flatten().repeat(v_data1.shape[0], x_data.shape[0])
            all_v_data = torch.where((domain >= v_data1[:, [0]]) & (domain <= v_data1[:, [1]]), 1.0, 0.0)
            all_v_data = all_v_data*v_data2
        elif labels:
            v_labels   = all_v_data[:,[3]]
            all_v_data = torch.cat((v_data1,v_data2,v_labels),axis=1) 
        else:
            all_v_data = torch.cat((v_data1,v_data2),axis=1)

        # shuffle data
        if shuffle:
            indices = torch.randperm(all_u_data.shape[0])
            all_u_data = all_u_data[indices]
            all_v_data = all_v_data[indices]
            return all_u_data, x_data, all_v_data, scale_fac, indices
        
    else:
        all_u_data, x_data, all_v_data, scale_fac = load_single_train(dataname,scaling,labels,full_v_data)
        if shuffle:
            indices = [i for i in range(all_u_data.shape[0])]
            return all_u_data, x_data, all_v_data, scale_fac, indices
        
    return all_u_data, x_data, all_v_data, scale_fac

"""
Load single test: when you have a single file
Test differs from train by the fact that the normalization is induced by the
parameters obtained by scale_fac of the train. Otherwise the network would 
have some information got from data that shouldn't know before the test phase.
"""
def load_single_test(dataname,scale_fac=None,scaling=None,labels=False,full_v_data=False):
    d         = sio.loadmat(dataname)
    u_data    = torch.tensor(d['vs']).float()
    x_data    = torch.tensor(d['tspan']).float()
    v_data1   = (torch.tensor(d['iapps'])[:,0:2]).float()   # pulse times
    v_data2   = (torch.tensor(d['iapps'])[:,[2]]).float()   # pulse intensities

    if scaling == "Default":
        u_data = scale_data_for_test(u_data,scale_fac[1],scale_fac[0],0.0,1.0) # scale_fac = [u_max,u_min,x_max,x_min,v_max1,v_min1,v_max2,v_min2]
        x_data = scale_data_for_test(x_data,scale_fac[3],scale_fac[2],0.0,1.0)
        v_data1 = scale_data_for_test(v_data1,scale_fac[5],scale_fac[4],0.0,1.0)
        v_data2 = scale_data_for_test(v_data2,scale_fac[7],scale_fac[6],0.0,1.0)
    elif scaling == "Gaussian":
        u_data = gaussian_scale_for_test(u_data,scale_fac[0],scale_fac[1]) # scale_fac = [u_mean,u_std,x_mean,x_std,v_mean1,v_std1,v_mean2,v_std2]
        x_data = gaussian_scale_for_test(x_data,scale_fac[2],scale_fac[3])
        v_data1 = gaussian_scale_for_test(v_data1,scale_fac[4],scale_fac[5])
        v_data2 = gaussian_scale_for_test(v_data2,scale_fac[6],scale_fac[7])
    elif scaling == "Mixed":
        u_data = gaussian_scale_for_test(u_data,scale_fac[0],scale_fac[1])
        x_data = scale_data_for_test(x_data,scale_fac[3],scale_fac[2],0.0,1.0)
        v_data1 = gaussian_scale_for_test(v_data1,scale_fac[4],scale_fac[5])
        v_data2 = gaussian_scale_for_test(v_data2,scale_fac[6],scale_fac[7])
    elif scaling == "Fourier": 
        # scaling used for FNO: u_data and x_data are not scaled
        v_data2 = gaussian_scale_for_test(v_data2,scale_fac[6],scale_fac[7])
        
    # Variant for v_data
    if full_v_data and labels:
        raise NotImplementedError("Full v and labels data is not implemented yet.")
    elif full_v_data:
        domain = x_data.flatten().repeat(v_data1.shape[0], x_data.shape[0])
        v_data = torch.where((domain >= v_data1[:, [0]]) & (domain <= v_data1[:, [1]]), 1.0, 0.0)
        v_data = v_data*v_data2
    elif labels:
        v_labels = (torch.tensor(d['iapps'])[:,[3]]).int()
        v_data   = torch.cat((v_data1,v_data2,v_labels),axis=1) 
    else:
        v_data = torch.cat((v_data1,v_data2),axis=1)

    return u_data, x_data.t(), v_data

def load_test(dataname,scale_fac=None,scaling=None,labels=False,full_v_data=False,shuffle=False):
    if isinstance(dataname,list):
        # Initialize empty lists to store the data
        all_u_data = []
        all_v_data = []
        x_data = None  # Initialize x_data

        for data in dataname:
            u_data, x_data, v_data = load_single_test(data,scale_fac,scaling,labels,full_v_data)
            # Append u_data and v_data to the lists
            all_u_data.append(u_data)
            all_v_data.append(v_data)

        # Stack the lists of tensors to form a single tensor
        all_u_data = torch.cat(all_u_data, dim=0)
        all_v_data = torch.cat(all_v_data, dim=0)

        # shuffle data
        if shuffle:
            indices = torch.randperm(all_u_data.shape[0])
            all_u_data = all_u_data[indices]
            all_v_data = all_v_data[indices]
            return all_u_data, x_data, all_v_data, indices
    else:
        all_u_data, x_data, all_v_data = load_single_test(dataname,scale_fac,scaling,labels,full_v_data)
        if shuffle:
            indices = [i for i in range(all_u_data.shape[0])]
            return all_u_data, x_data, all_v_data, indices

    return all_u_data, x_data, all_v_data


""" main for test """
if __name__=="__main__":
    dataname = ["dataset/datasetHH_test_0peaks.mat",
                "dataset/datasetHH_test_1peaks.mat",
                "dataset/datasetHH_test_2peaks.mat",
                "dataset/datasetHH_test_3peaks.mat",
                "dataset/datasetHH_test_4peaks.mat",
                "dataset/datasetHH_test_5peaks.mat",
                "dataset/datasetHH_test_6peaks.mat"]
    
    dataname_test = ["dataset/datasetHH_test_0peaks_test.mat",
                "dataset/datasetHH_test_1peaks_test.mat",
                "dataset/datasetHH_test_2peaks_test.mat",
                "dataset/datasetHH_test_3peaks_test.mat",
                "dataset/datasetHH_test_4peaks_test.mat",
                "dataset/datasetHH_test_5peaks_test.mat",
                "dataset/datasetHH_test_6peaks_test.mat"]
    
    dataname = "dataset/datasetHH_test_5peaks.mat"
               
    dataname_test = "dataset/datasetHH_test_5peaks_test.mat"
                
    scaling     = "Default"
    labels      = True
    full_v_data = False
    shuffle     = True

    u_train, x_train, v_train, scale_fac, indices = load_train(dataname,scaling,labels,full_v_data,shuffle)
    #u_test, x_test, v_test, indices = load_test(dataname_test,scale_fac,scaling,labels,full_v_data,shuffle)
    u_test, x_test, v_test = load_test(dataname_test,scale_fac,scaling=scaling)
    print("Effettuato")
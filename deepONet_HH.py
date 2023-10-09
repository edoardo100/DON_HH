import scipy.io as sio
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
from jax import random
from jax import jit, value_and_grad
import time
from tensorboardX import SummaryWriter
# for test launcher interface
import os
import yaml
import argparse

# Define command-line arguments
parser = argparse.ArgumentParser(description="Learning Hodgkin-Huxley model with Fourier Neural Operator")
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
scaling       = config["scaling"]
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

#### Functions
def scale_data(data, min_value, max_value):
    data_min = jnp.min(data)
    data_max = jnp.max(data)
    # Apply the linear transformation
    scaled_data = (max_value - min_value) * (data - data_min) / (data_max - data_min) + min_value
    return data_min, data_max, scaled_data

def unscale_data(scaled_data, original_min, original_max):
    # Apply the inverse linear transformation
    unscaled_data = (scaled_data - original_min) / (original_max - original_min)
    # Map the unscaled data back to the original range
    unscaled_data = unscaled_data * (original_max - original_min) + original_min
    return unscaled_data

def gaussian_scale(data):
    mean = jnp.mean(data)
    std_dev = jnp.std(data)
    scaled_data = (data - mean) / std_dev
    return mean, std_dev, scaled_data

def inverse_gaussian_scale(scaled_data, original_mean, original_std_dev): 
    unscaled_data = scaled_data * original_std_dev + original_mean
    return unscaled_data

def load_dataset(dataname,split,scaling):
    d         = sio.loadmat(dataname)
    u_data    = jnp.array(d['vs'])
    x_data    = jnp.array(d['tspan'])
    v_data1   = jnp.array(d['iapps'])[:,0:2] # pulse times
    v_data2   = jnp.array(d['iapps'])[:,2]   # pulse intensities
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
        x_mean, x_std, x_data = scale_data(x_data,0.0,1.0)
        v_mean1, v_std1, v_data1 = gaussian_scale(v_data1)
        v_mean2, v_std2, v_data2 = gaussian_scale(v_data2)
        scale_fac = [u_mean,u_std,x_max,x_min,v_mean1,v_std1,v_mean2,v_std2]
    v_data = jnp.concatenate((v_data1,v_data2.reshape(-1,1)),axis=1)
    u_train, x_train, v_train = u_data[:split], x_data.transpose(), v_data[:split]
    u_test, x_test, v_test = u_data[split:], x_data.transpose(), v_data[split:]
    return scale_fac, u_train, x_train, v_train, u_test, x_test, v_test

key = random.PRNGKey(1234)
initializer = jax.nn.initializers.glorot_normal()

def hyper_initial(layers,adapt_actfun=True):
    L = len(layers)
    W = []
    b = []
    for l in range(1, L):
        in_dim = layers[l-1]
        out_dim = layers[l]
        std = jnp.sqrt(2.0/(in_dim+out_dim))
        weight = initializer(key, (in_dim, out_dim), jnp.float32)*std
        bias = initializer(key, (1, out_dim), jnp.float32)*std
        W.append(weight)
        b.append(bias)
    if adapt_actfun:
        a = jax.nn.initializers.ones(key, (L-1, 1), jnp.float32)*0.1
        c = jax.nn.initializers.ones(key, (L-1, 1), jnp.float32)*0.1
        return W, b, a, c
    return W, b

def fnn_B_adapt(X, W, b, a, c):
    inputs = X
    L = len(W)
    for i in range(L-1):
        outputs = jnp.dot(inputs, W[i]) + b[i]
        inputs = jnp.sin(10*a[i]*outputs+c[i])  
    Y = jnp.dot(inputs, W[-1]) + b[-1]     
    return Y

def fnn_B(X, W, b):
    inputs = X
    L = len(W)
    for i in range(L-1):
        outputs = jnp.dot(inputs, W[i]) + b[i]
        inputs = jnp.tanh(outputs)  
    Y = jnp.dot(inputs, W[-1]) + b[-1]     
    return Y

def fnn_T_adapt(X, W, b, a, c):
    inputs = X
    L = len(W)
    for i in range(L-1):
        outputs = jnp.dot(inputs, W[i]) + b[i]      
        inputs = jnp.tanh(10*a[i]*outputs+c[i])
        Y = jnp.dot(inputs, W[-1]) + b[-1]     
    return Y

def fnn_T(X, W, b):
    inputs = X
    L = len(W)
    for i in range(L-1):
        outputs = jnp.dot(inputs, W[i]) + b[i]
        inputs = jnp.tanh(outputs)                   # inputs to the next layer
    Y = jnp.dot(inputs, W[-1]) + b[-1]     
    return Y

def predict_vanilla(params,data):
    v, x = data
    W_branch, b_branch, W_trunk, b_trunk = params
    u_out_branch = fnn_B(v, W_branch, b_branch)         # predict on branch
    u_out_trunk  = fnn_T(x, W_trunk, b_trunk)            # predict on trunk
    u_pred = jnp.einsum('ij,kj->ik',u_out_branch, u_out_trunk)
    return u_pred

def predict_adapt(params,data):
    v, x = data
    W_branch, b_branch, a_b, c_b, W_trunk, b_trunk, a_t, c_t = params
    u_out_branch = fnn_B_adapt(v, W_branch, b_branch, a_b, c_b)
    u_out_trunk  = fnn_T_adapt(x, W_trunk, b_trunk, a_t, c_t)
    u_pred = jnp.einsum('ij,kj->ik',u_out_branch, u_out_trunk)
    return u_pred

if adapt_actfun:
    predict = predict_adapt
else:
    predict = predict_vanilla

# Loss = MSE ### change to other losses
def loss(params, data, u):
    u_preds = predict(params, data)
    mse = jnp.mean((u_preds.flatten() - u.flatten())**2)
    return mse

# schedule learning rate
if scheduler == "None":
    lr_sched = lr
elif scheduler == "Exp":
    eps      = 10000
    lr_sched = optimizers.exponential_decay(lr, eps, 0.5)
else:
    raise ValueError("Scheduler not stated")
opt_init, opt_update, get_params = optimizers.adam(lr_sched)

@jit
def update(params, data, u, opt_state):
    """ Compute the gradient for a batch and update the parameters """
    value, grads = value_and_grad(loss)(params, data, u)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value

if __name__ == '__main__':
    writer = SummaryWriter(log_dir = name_log_dir)
    scale_fac, u_train, x_train, v_train, \
    u_test, x_test, v_test = load_dataset(dataname,split,scaling)
    
    layers_f = [u_dim] + inner_layer_b + [G_dim]   # Branch Net
    layers_x = [x_dim] + inner_layer_t + [G_dim]   # Trunk Net

    #### Initialize hyperparams
    if adapt_actfun:
        W_branch, b_branch, a_branch, c_branch = hyper_initial(layers_f,adapt_actfun)
        W_trunk,  b_trunk,  a_trunk,  c_trunk  = hyper_initial(layers_x,adapt_actfun)
        params    = [W_branch, b_branch, a_branch, 
                    c_branch, W_trunk, b_trunk, 
                    a_trunk, c_trunk]
    else:
        W_branch, b_branch = hyper_initial(layers_f,adapt_actfun)
        W_trunk,  b_trunk = hyper_initial(layers_x,adapt_actfun)
        params    = [W_branch, b_branch,
                     W_trunk, b_trunk]
    par_tot = sum([i.flatten().shape[0] for j in params for i in j])
    print("Total Operator Network parameters number is: ", par_tot)
    writer.add_text("Parameters", 'Total parameters number: ' + str(par_tot), 0)
    #### Main training loop
    train_loss, test_loss = [], []
    start_time = time.time()
    opt_state = opt_init(params)

    for epoch in range(epochs):
        params, opt_state, loss_val = update(params, [v_train, x_train], u_train, opt_state)
        train_loss.append(loss_val)
        if epoch % 100 ==0:
            epoch_time = time.time() - start_time
            u_train_pred = predict(params, [v_train, x_train])
            err_train = \
                  jnp.mean(jnp.linalg.norm(u_train - u_train_pred, 2, axis=1)/\
                           jnp.linalg.norm(u_train , 2, axis=1))
            u_test_pred = predict(params, [v_test, x_test])
            err_test = \
                    jnp.mean(jnp.linalg.norm(u_test - u_test_pred, 2, axis=1)/\
                             jnp.linalg.norm(u_test , 2, axis=1))
            test_loss_val = loss(params, [v_test, x_test], u_test)
            test_loss.append(test_loss_val)
            print("Epoch {} | T: {:0.6f} | Train MSE: {:0.3e} | \
            Test MSE: {:0.3e} | Train L2: {:0.3e} | Test L2: {:0.3e}"\
            .format(epoch, epoch_time, loss_val, 
                    test_loss_val, err_train, err_test))
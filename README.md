# DNO_HH

Learning Hodgkin-Huxley model with DeepONet

File and folders: 
* deepONet_HH_pytorch.py : main
* default_params_new.yml : config yml file template
* src : local modules for DeepONet_HH_pytorch.py
* legacy_files : old files and prototipes
## Usage:

`python3 deepONet_HH_pytorch.py --config_file <yml_file.yml>`

### To recover trained models:

`python3 recover_model.py --config_file <yml_file.yml>`

please change lines in the code indicating the new indexes to predict (idx) and the model path (modelname)

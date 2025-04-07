import pdb
import torch  
import torchprofile

# Function to calculate FLOPs 
def calculate_flops(model, model_input_arrays, inputs_list):
    model.eval() 

    tmp = [[], [], [], []] 
    if "dvs" in inputs_list: tmp[0] = model_input_arrays[0]
    if "rgb" in inputs_list: tmp[1] =  model_input_arrays[1]
    if "imu_a" in inputs_list: tmp[2] =  model_input_arrays[2]
    if "imu_g" in inputs_list: tmp[3] =  model_input_arrays[3]

    macs = torchprofile.profile_macs(model, tuple(tmp))
    flops = macs * 2  # FLOPs are typically 2x MACs
    return flops, macs
 
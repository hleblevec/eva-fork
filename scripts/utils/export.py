import torch 
import os
import pdb
import subprocess
import yaml

def get_max_epochs_and_steps(num_samples, batch_size, max_steps, max_epochs):
    if max_steps is None: 
        return max_epochs, num_samples*max_epochs//batch_size
    elif max_epochs is None:
        return max_steps*batch_size//num_samples, max_steps
    
def install_package(package_name, version=""):
    try:
        __import__(package_name)
    except ImportError:
        subprocess.check_call(["pip", "install", package_name+"=="+version])

def rename_layers(onnx_model):
    for node in onnx_model.graph.node:
        if '/' in node.name:
            node.name = re.sub(r'/', '_', node.name)
        for i, input_name in enumerate(node.input):
            if '/' in input_name:
                node.input[i] = re.sub(r'/', '_', input_name)
        for i, output_name in enumerate(node.output):
            if '/' in output_name:
                node.output[i] = re.sub(r'/', '_', output_name)
    return onnx_model

def save_model_to(model, extension, path):
    if extension == ".onnx": 
        install_package("onnx")
        import onnx
        model.to_onnx(os.path.join(path,"model.onnx"), export_params=True) 
    elif extension == ".h5":  
        install_package("tensorflow", "2.15")
        install_package("keras")
        install_package("nobuco")
        import nobuco
        from nobuco import ChannelOrder
        from keras.models import save_model

        input_args = []
        for tensor_input in model.example_input_array:
            dummy = torch.rand(size=tensor_input.shape)
            input_args.append(dummy)
 
        pytorch_module = model.eval()
        keras_model = nobuco.pytorch_to_keras(
            pytorch_module,
            args=input_args, kwargs=None,
            inputs_channel_order=ChannelOrder.TENSORFLOW,
            outputs_channel_order=ChannelOrder.TENSORFLOW
        )
        save_model(keras_model, os.path.join(path, "model.h5"))  

    elif extension == ".pth":
        torch.save(model.model.state_dict(), os.path.join(path,"model.pth"))


def save_configs(configs, out_dir, filename='config.yaml'):
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, filename)
    with open(config_path, 'w') as file:
        yaml.dump(configs, file, default_flow_style=False)


def load_configs(config_path):
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    return configs
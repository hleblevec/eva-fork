import torch
import torch.nn as nn
import pdb 

from .layers import SelfAttentionBlock, ConvBlock, LinearBlock, ResidualBlock, xilinx_leaky_relu_neg_slope

class Eva1Branch(nn.Module):
    def __init__(self, in_dim, out_dim, configs, bias, verbose=False):
        super().__init__() 
        self.branch = nn.Sequential(
            ConvBlock(in_dim, 48, configs, stride=1, padding=0, bias=bias),
            nn.AvgPool2d(2,2),
            nn.Dropout(0.5),
            
            ConvBlock(48, 56, configs, stride=1, padding=0, bias=bias),
            nn.Dropout(0.5),
            
            ConvBlock(56, 64, configs, stride=1, padding=0, bias=bias),
            nn.AvgPool2d(2,2),
            nn.Dropout(0.5),
            
            ConvBlock(64, 72, configs, stride=1, padding=0, bias=bias),
            nn.Dropout(0.5),
            
            ConvBlock(72, 84, configs, stride=1, padding=0, bias=bias),
            nn.AvgPool2d(2,2),
            nn.Dropout(0.5),
            
            ConvBlock(84, 96, configs, stride=1, padding=0, bias=bias),
            nn.AvgPool2d(2,2),
            nn.Dropout(0.5),
            nn.Flatten(),
            
            LinearBlock(384, 128, configs, activation=nn.LeakyReLU(xilinx_leaky_relu_neg_slope), bias=bias),
            nn.Dropout(0.5),
            
            LinearBlock(128, out_dim, configs, activation=nn.LeakyReLU(xilinx_leaky_relu_neg_slope), bias=bias),
            nn.Dropout(0.5),
        )
        self.verbose = verbose

    def forward(self, x):
        for layer in self.branch:
            x = layer(x)
            if self.verbose : print(f"Layer {layer.__class__.__name__}: Output shape {x.shape}")
        return x


class Eva2Branch(nn.Module):
    def __init__(self, in_dim, out_dim, configs, bias, verbose=False):
        super().__init__() 
        
        self.branch = nn.Sequential(
            
            # input encoder
            ConvBlock(in_dim, 8, configs, bias=bias, stride=1), 

            # encoder - 3 residuals
            ResidualBlock(8, 8, configs, bias=bias),
            ConvBlock(8, 16, configs, bias=bias, stride=2), 
            nn.Dropout(0.2),
                 
            ResidualBlock(16, 16, configs, bias=bias),
            ConvBlock(16, 32, configs, bias=bias, stride=2), 
            nn.Dropout(0.2),
            
            ResidualBlock(32, 32, configs, bias=bias),
            ConvBlock(32, 64, configs, bias=bias, stride=2), 
            nn.Dropout(0.2),
            
            # bottleneck
            ResidualBlock(64, 64, configs, bias=bias), 
            
            # decoder - 3 residuals
            ConvBlock(64, 32, configs, bias=bias, stride=2),  
            ResidualBlock(32, 32, configs, bias=bias), 
            nn.Dropout(0.2), 
            
            ConvBlock(32, 16, configs, bias=bias, stride=2), 
            ResidualBlock(16, 16, configs, bias=bias), 
            nn.Dropout(0.2), 

            ConvBlock(16, 8, configs, bias=bias, stride=2), 
            ResidualBlock(8, 8, configs, bias=bias), 
            nn.Dropout(0.2), 
            
            # predictor
            nn.Flatten(),
            LinearBlock(32, out_dim, configs, bias=bias),
            nn.Dropout(0.2)
            )
        
        self.verbose = verbose 

    def forward(self, x):
        for layer in self.branch:
            x = layer(x)
            if self.verbose : print(f"Layer {layer.__class__.__name__}: Output shape {x.shape}")
        return x

versions ={
    1: Eva1Branch,
    2: Eva2Branch
}

# EvaNet
class EvaNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        # I/O Dimension
        self.in_dim_rgb = 3 * config["rgb_windows"]
        self.in_dim_dvs = config["event_polarities"] * config["event_windows"]

        self.event_windows = config["event_windows"]
        self.inputs_list = config["inputs_list"]
        self.outputs_list = config["outputs_list"]  
        self.out_dim = 3

        # Layers
        hidden_dim = 16
        self.model_rgb = versions[config["model_version"]](self.in_dim_rgb, hidden_dim, config, bias=config["bias"]) if "rgb" in self.inputs_list else None
        self.model_dvs = versions[config["model_version"]](self.in_dim_dvs, hidden_dim, config, bias=config["bias"]) if "dvs" in self.inputs_list else None
        
        if self.model_rgb and self.model_dvs:
            self.attention_block = SelfAttentionBlock(hidden_dim)
            self.predictor = LinearBlock(hidden_dim*2, self.out_dim, config, activation=None, bias=config["bias"])
        else:
            self.predictor = LinearBlock(hidden_dim, self.out_dim, config, activation=None, bias=config["bias"])

    def forward(self, i_dvs=[], i_rgb=[], i_imu_g=[], i_imu_a=[]):

        is_rgb_available = (i_rgb != []) and (not (i_rgb == -1).all()) 

        if i_dvs != [] and is_rgb_available:
            out_rgb = self.model_rgb(i_rgb)
            out_dvs = self.model_dvs(i_dvs)
            fused_features = torch.stack([out_rgb, out_dvs], dim=1) 
            attention_out = self.attention_block(fused_features) 
            #attention_out = attention_out.mean(dim=1)
            attention_out = attention_out.flatten(1)
            out_tot = self.predictor(attention_out)
            return out_tot

        elif is_rgb_available:
            out_rgb = self.model_rgb(i_rgb)
            return self.predictor(out_rgb)

        elif i_dvs != []:
            out_dvs = self.model_dvs(i_dvs)
            return self.predictor(out_dvs)

        else:
            raise ValueError("At least one input modality must be provided.")

if __name__ == '__main__':
    config ={
        "inputs_list":["dvs"],
        "outputs_list":["delta_drone_2_point_of_collision_yzt"],
        "event_windows": 1,
        "event_polarities": 1,
        "rgb_windows": 1,
        "imu_windows": 1,
        "dvs_res": [80, 80], 
        "batch_size": 1,
        "block_method": "normal",
        "precision": 32,
        "bias": True
    }

    # input tensors
    dvs_tensor, rgb_tensor = [], []
    if "dvs" in config["inputs_list"]:
        dvs_tensor = torch.randn((config["batch_size"], config["event_windows"] * config["event_polarities"], config["dvs_res"][0], config["dvs_res"][1]))
    if "rgb" in config["inputs_list"]:
        rgb_tensor = torch.randn((config["batch_size"], config["rgb_windows"] * 3, config["dvs_res"][0], config["dvs_res"][1]))
    imu_a_tensor = torch.randn((config["batch_size"], config["imu_windows"], 3))
    imu_g_tensor = torch.randn((config["batch_size"], config["imu_windows"], 3))

    model = EvaNet(config)
    example_output = model(i_dvs=dvs_tensor, i_rgb=rgb_tensor) 

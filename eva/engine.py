import wandb
import os
import torch
import pdb
import numpy as np
import pytorch_lightning as pl 
from pytorch_lightning.utilities.model_summary import ModelSummary

from .plots.plotter import Plotter 

from .nn.model import EvaNet
from .nn.utils.utils import calculate_flops
from .nn.quantizer import lsq_quantization

from .data.transforms import move_tensor_dict_to_device
from .metrics.loss import PositionVelocityLoss, AutomaticWeightedLoss

def define_example_arrays(config):
    # input tensors
    dvs_tensor, rgb_tensor, imu_a_tensor, imu_g_tensor = [], [], [], []
    if "dvs" in config["inputs_list"]:
        dvs_tensor = 2 * torch.rand((1, config["event_windows"]*config["event_polarities"], config["dvs_res"][0], config["dvs_res"][1])) - 1
    if "rgb" in config["inputs_list"]:
        rgb_tensor = 2 * torch.rand((1, config["rgb_windows"]*3, config["dvs_res"][0], config["dvs_res"][1])) - 1
    if "imu" in config["inputs_list"]:
        imu_a_tensor = torch.randn((1, config["imu_windows"], 3)) 
        imu_g_tensor = torch.randn((1, config["imu_windows"], 3)) 

    # output tensors
    output_tensor = torch.randn((1, 3)) 
    return (dvs_tensor, rgb_tensor, imu_a_tensor, imu_g_tensor), (output_tensor)

class ModelEngine(pl.LightningModule):
    def __init__(self, config, q_config):
        super().__init__()

        self.config = config
        self.model = EvaNet(config) 
        
        if config["bias"] == False:
            for name, module in self.model.named_modules():
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias = None

        self.multitask_uncertainty_loss = False
        self.loss_function = PositionVelocityLoss(multitask_uncertainty_loss=self.multitask_uncertainty_loss) 
        self.learning_rate = config["learning_rate"]

        self.example_input_array, self.example_output_array = define_example_arrays(config)   
        self.plotter = { 
            "train":Plotter(config=config),
            "val":Plotter(config=config),
            "test":Plotter(config=config)
        } 
        self.outputs_list = config["outputs_list"]
        self.max_epochs = self.config["max_epochs"] 

        # Prepare for quantization
        if self.config["precision"] < 32 and self.config["precision"] > 1:
            if q_config["quant_method"] == "lsq":
                self.model = lsq_quantization(self.model, q_config)

    def on_train_start(self):
        summary = ModelSummary(self, max_depth=-1)
        self.log("complexity/model_size", summary.model_size)
        self.log("complexity/total_parameters", summary.total_parameters)
        self.model = self.model.to(self.device)
 
        example_input_array = tuple(item.to(self.device) if item != [] else [] for item in self.example_input_array)
        flops, macs = calculate_flops(self.model, example_input_array, self.config["inputs_list"]) 
        self.log("complexity/flops", flops)
        print(f"FLOPs: {flops:.2e}")
        self.log("complexity/macs", macs) 
        print(f"MACs: {macs:.2e}")

    def forward(self, i_dvs=[], i_rgb=[], i_imu_g=[], i_imu_a=[]):   
        output_tensor = self.model(i_dvs, i_rgb, i_imu_g, i_imu_a) 
        if "delta_drone_2_point_of_collision_yzt" in self.outputs_list:
            return {"delta_drone_2_point_of_collision_yzt": output_tensor}  
        elif "delta_ball_2_point_of_collision_yzt" in self.outputs_list:
            return {"delta_ball_2_point_of_collision_yzt": output_tensor}   
        elif "delta_ball_2_drone__3d_xyz" in self.outputs_list:
            return {"delta_ball_2_drone__3d_xyz": output_tensor}   
        
    def replace_mp_with_standard(self):
        self.model.replace_mp_with_standard()

    def load_weights(self, path_to_ckpt): 
        state_dict = torch.load(path_to_ckpt, weights_only=False)["state_dict"] 
        self.load_state_dict(state_dict, strict=False)
        print("Loaded Weights") 

    def logging_step(self, loss, outputs, targets, batch, dataset_name):   
        self.plotter[dataset_name].update_stats(targets=targets, outputs=outputs, batch=batch, dataset_name=dataset_name) 
        for key in loss.keys():
            self.log(f"{dataset_name}/"+str(key), float(loss[key]), on_epoch=True) 

    def training_step(self, batch): 
        inputs = move_tensor_dict_to_device(batch["inputs"], self.device)
        targets = move_tensor_dict_to_device(batch["targets"], self.device)  
        outputs = self.forward(**inputs) 
        loss = self.loss_function(outputs, targets)
        
        if self.scheduler is not None:
            self.log("trainer/learning_rate", self.scheduler.get_last_lr()[0], on_epoch=True)

        if self.multitask_uncertainty_loss:
            self.log("trainer/w_0", self.loss_function.loss_automatic_weight.params[0]) 
            self.log("trainer/w_1", self.loss_function.loss_automatic_weight.params[1]) 
            
        self.logging_step(loss=loss, targets=targets, outputs=outputs, batch=batch, dataset_name="train")  
        return loss["total_loss"]

    def validation_step(self, batch, idx):  
        inputs = move_tensor_dict_to_device(batch["inputs"], self.device)
        targets = move_tensor_dict_to_device(batch["targets"], self.device)
        outputs = self.forward(**inputs)
        loss = self.loss_function(outputs, targets)   
        self.logging_step(loss=loss, targets=targets, outputs=outputs, batch=batch, dataset_name="val")  
        return loss["total_loss"]

    def test_step(self, batch, idx, dataloader_idx=0): 
        inputs = move_tensor_dict_to_device(batch["inputs"], self.device)
        targets = move_tensor_dict_to_device(batch["targets"], self.device) 
        outputs = self.forward(**inputs)
        loss = self.loss_function(outputs, targets)
        self.logging_step(loss=loss, targets=targets, outputs=outputs, batch=batch, dataset_name="test")  
        return loss["total_loss"]

    def configure_optimizers(self): 
        param_groups = [
            {"params": self.model.parameters(), "lr": self.learning_rate}, 
            {"params": self.loss_function.parameters(), "lr": self.learning_rate * 0.1}
        ]
    
        self.optimizer = torch.optim.Adam(param_groups, lr=self.learning_rate, weight_decay=1e-5)
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)
        
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'monitor': 'val/total_loss'
            }
        }

    def on_epoch_end_logs(self, name):
        stats_dict = self.plotter[name].compute_stats()    

        # Initialize lists to store true labels and predictions
        columns = ["tn", "fp", "fn", "tp"]
        tn, fp, fn, tp = stats_dict["metrics"]["conf_matrix"].ravel() 
        test_table = wandb.Table(data=[[tn, fp, fn, tp]], columns=columns)

        wandb.log({f"{name}/confusion_matrix": test_table})
        self.log(f"{name}/precision", stats_dict["metrics"]["precision"])
        self.log(f"{name}/recall", stats_dict["metrics"]["recall"])
        self.log(f"{name}/f1", stats_dict["metrics"]["F1"])
        self.log(f"{name}/accuracy", stats_dict["metrics"]["accuracy"])  

        if self.outputs_list[0] == "delta_drone_2_point_of_collision_yzt":
            self.log(f"{name}/delta_yz_mu_mm", stats_dict["delta_yz"]["mean"], on_epoch=True)  
            self.log(f"{name}/delta_yz_std_mm", stats_dict["delta_yz"]["std"], on_epoch=True)  
            self.log(f"{name}/delta_t_mu_ms", stats_dict["delta_t"]["mean"], on_epoch=True)  
            self.log(f"{name}/delta_t_std_ms", stats_dict["delta_t"]["std"], on_epoch=True)   
        elif self.outputs_list[0] == "delta_ball_2_drone__3d_xyz":
            self.log(f"{name}/distance_mu_mm", stats_dict["distances"]["mean"], on_epoch=True)   
            self.log(f"{name}/distance_std_mm", stats_dict["distances"]["std"], on_epoch=True)   

        self.plotter[name].reset()  

    def on_train_epoch_end(self):
        self.on_epoch_end_logs("train")  
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        filename = "val_" + str(self.current_epoch) + ".mp4" 
        if self.current_epoch >= self.max_epochs - 1:
            self.plotter["val"].plot(os.path.join(self.config["out_dir"], filename)) 
        self.on_epoch_end_logs("val")
        torch.cuda.empty_cache()  

    def on_validation_epoch_start(self): 
        self.plotter["val"].reset()
        torch.cuda.empty_cache()  

    def on_train_epoch_start(self): 
        self.plotter["train"].reset()

    def on_test_epoch_end(self):
        self.on_epoch_end_logs("test") 
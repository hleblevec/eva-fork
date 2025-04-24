import os
import fire
import pdb 
from dotenv import load_dotenv
load_dotenv()
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# PyTorch imports
import torch 
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import StochasticWeightAveraging, EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything
seed_everything(1234, workers=True)

# Custom imports
from eva.engine import ModelEngine
from eva.nn.utils.callbacks import VanishingGradientChecker, WeightHealthMonitor, WeightMagnitudeDistribution
from eva.data.module import ABCDataModule
from .utils.export import save_configs, save_model_to, load_configs, get_max_epochs_and_steps
    
def main(   
            name='test123', 
            ckpt_dir=None,  
            inputs_list=["dvs"],
            frequency="dvs",
            max_steps=None, 
            max_epochs=100,
            event_polarities=1,
            event_polarities_mode="substract",
            event_windows=1,
            rgb_windows=1,
            batch_size=32,
            event_dt_ms=20, 
            device=1,
            precision=32,
            block_method="normal",
            model_version=1,
            learning_rate=1e-4,
            output="delta_drone_2_point_of_collision_yzt",
            dvs_res=[80,80], 
            test_splits_list=[1]
        ):

    outpath = os.path.join(os.getenv("OUTPUT_PATH"), 'results', name) 
    os.makedirs(outpath, exist_ok=True) 
    
    # Data configurations
    base_config = { 
        "outputs_list": [output],
        "inputs_list": inputs_list
    }  
    data_config = {
        "in_dir": os.getenv("ABCD_DATA_PATH"),
        "num_workers": 1,
        "batch_size": batch_size,
        "dvs_res": dvs_res,
        "imu_windows": 1,
        "rgb_windows": rgb_windows,
        "event_dt_ms": event_dt_ms,
        "event_windows": event_windows,
        "event_polarities": event_polarities,
        "event_polarities_mode": event_polarities_mode,
        "event_accumulation": "addition",
        "event_decay_constant": 0.9,
        "frequency": frequency,
        "overwrite": False,
        "outputs_list": base_config["outputs_list"],
        "inputs_list": base_config["inputs_list"],
        "test_splits_list": test_splits_list,
    } 
    exp_config = {
        "out_dir": outpath, 
        "swa": False, 
        "early_stopping": True
    }
    
    # Initialize logging
    tensorboard_logger = TensorBoardLogger(name="tensorboard", save_dir=exp_config["out_dir"])
    wandb_logger = WandbLogger(project="eva", name=name, save_dir=exp_config["out_dir"], mode="run")
    torch.set_float32_matmul_precision('medium')
    
    # Loading callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/total_loss",
        dirpath=os.path.join(outpath, "checkpoints"),
        filename="best-checkpoint",
        save_top_k=1,
        mode="min")
    callbacks = [VanishingGradientChecker(), WeightMagnitudeDistribution(), checkpoint_callback]
    if exp_config["swa"]: callbacks.append(StochasticWeightAveraging(swa_lrs=learning_rate)) 
    if exp_config["early_stopping"]: callbacks.append(EarlyStopping(monitor='val/total_loss', patience=10, verbose=True, mode='min'))
 
    # Model configuration
    model_config = {
        "bias": False if block_method == "mp" else True,
        "model_version": model_version,
        "precision": precision,
        "block_method": block_method,
        "w_pos": 1,
        "w_vel": 0, 
        "w_ttc": 0, 
        "learning_rate": learning_rate,
        "out_dir": exp_config["out_dir"],
        "max_steps": max_steps,
        "max_epochs": max_epochs,
        "batch_size": data_config["batch_size"],
        "dvs_res": data_config["dvs_res"],
        "rgb_windows": data_config["rgb_windows"],
        "event_windows": data_config["event_windows"],
        "event_polarities": data_config["event_polarities"],
        "event_accumulation": data_config["event_accumulation"],
        "event_decay_constant": data_config["event_decay_constant"],
        "imu_windows": data_config["imu_windows"],
        "outputs_list": base_config["outputs_list"],
        "inputs_list": base_config["inputs_list"],  
    } 
    q_config = {
        "a_bit": precision,
        "w_bit": precision,
        "all_positive": False,
        "per_channel": True,
        "quant_inference": True,
        "batch_init": data_config["batch_size"],
        "quant_method": "lsq"
    }

    # Loading checkpoint configs
    if ckpt_dir is not None:
        config = load_configs(os.path.join(ckpt_dir, "config.yaml"))
        base_config = config['base_config']
        data_config = config['data_config']
        exp_config = config['exp_config']
        model_config = config['model_config']
        q_config = config['q_config'] 

    # Initialize data and training module
    data_module = ABCDataModule(config=data_config) 
    data_module.setup(stage="fit")
    num_samples = len(data_module.train_dataset)
    max_epochs, max_steps = get_max_epochs_and_steps(num_samples, data_config["batch_size"], max_steps, max_epochs)
    model_config["max_epochs"], model_config["max_steps"] = max_epochs, max_steps 
    model = ModelEngine(model_config, q_config)

    if ckpt_dir is not None:
        model.load_weights(os.path.join(ckpt_dir, "model.ckpt"))  
    
    trainer = pl.Trainer(
        max_steps=max_steps, 
        max_epochs=max_epochs, 
        callbacks=callbacks,
        accelerator='gpu',  
        num_sanity_val_steps=0,  
        log_every_n_steps=30, 
        devices=[device], 
        deterministic=True,
        gradient_clip_val=1.0,
        logger=[tensorboard_logger, wandb_logger]
        )

    # Save configs
    save_configs({
        'base_config': base_config,
        'data_config': data_config,
        'exp_config': exp_config,
        'model_config': model_config,
        'q_config': q_config
    }, out_dir=exp_config["out_dir"])

    # Train the model
    model.train()
    trainer.fit(model, datamodule=data_module)

    # Run validation and testing 
    if max_epochs > 1: 
        model.load_weights(checkpoint_callback.best_model_path)
    trainer.validate(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module) 

    # Export model 
    save_model_to(model, extension=".pth", path=exp_config["out_dir"]) 
    save_model_to(model, extension=".onnx", path=exp_config["out_dir"]) 

if __name__ == "__main__":
    fire.Fire(main)

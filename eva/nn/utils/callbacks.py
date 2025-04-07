import wandb, pdb
import numpy as np
from pytorch_lightning.callbacks import Callback

class PrintAveragedMetricsCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        print("\nAveraged metrics at the end of training:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

class WeightMagnitudeDistribution(Callback):
    def on_after_backward(self, trainer, pl_module):
        # Access the TensorBoard logger
        writer = trainer.logger.experiment  

        # Collect weight magnitudes per layer
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                magnitudes = param.data.abs().view(-1).cpu().numpy()
                
                # Check the range of magnitudes to determine number of bins
                data_range = np.ptp(magnitudes)  # Peak-to-peak (max - min) range of data
                num_bins = 64  # Default number of bins

                # If the data range is smaller than the number of bins, adjust the number of bins
                if data_range <= num_bins:
                    num_bins = max(1, int(data_range))  # Set num_bins to the data range if it's smaller
                
                # Log weight magnitudes as histogram per layer to TensorBoard
                writer.add_histogram(f"weights/magnitude_distribution/{name}", magnitudes, global_step=trainer.global_step)

class VanishingGradientChecker(Callback):
    def on_after_backward(self, trainer, pl_module):
        # Log gradient norms for each parameter
        gradient_norms = {}
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()  # L2 norm
                gradient_norms[name] = grad_norm
        
        # Log gradient norms
        # trainer.logger.log_metrics({"gradients/grad_norms": gradient_norms}, step=trainer.global_step)

        # Optionally, log vanishing gradient warning if necessary
        for name, grad_norm in gradient_norms.items():
            if grad_norm < 1e-6:  # Threshold for considering a gradient "vanished"
                trainer.logger.log_metrics(
                    {f"vanishing_grad_warning/{name}": grad_norm}, step=trainer.global_step
                )

class WeightHealthMonitor(Callback):
    def __init__(self, log_interval=100):
        super().__init__()
        self.log_interval = log_interval  # Log every `log_interval` steps

    def on_after_backward(self, trainer, pl_module):
        weight_stats = {}
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                #weight_stats[f"weights/{name}/mean"] = param.data.mean().item()
                #weight_stats[f"weights/{name}/std"] = param.data.std().item()
                weight_stats[f"weights/{name}/max"] = param.data.max().item()
                weight_stats[f"weights/{name}/min"] = param.data.min().item()

        trainer.logger.log_metrics(weight_stats, step=trainer.global_step)

import pdb, os, torch, yaml, copy
from torch.utils.data import DataLoader
from collections import defaultdict, deque
import numpy as np
import math
from dotenv import load_dotenv
from scipy.ndimage import median_filter, gaussian_filter1d

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    accuracy_score,
)

from eva.metrics.collisions import CollisionStats
from eva.nn.model import EvaNet
from eva.data.dataset import ABCDataset
from eva.data.utils.utils import load_yaml_from_txt
from eva.data.transforms import TensorToNumpy, get_transforms, move_tensor_dict_to_device

from pytorch_lightning import seed_everything
seed_everything(1234, workers=True)

device = torch.device("cuda")
load_dotenv()

# configs and paths 
base_path = os.path.join(os.getenv("OUTPUT_PATH"), 'results', "dvs-model") 
config = load_yaml_from_txt(os.path.join(base_path, "config.yaml"))

# load model 
model = EvaNet(config["model_config"])
model.load_state_dict(torch.load(os.path.join(base_path, "model.pth")))
model = model.to(device)
model.eval() 

# load the val dataset
dataset_name = "test_dpu"
data_config = config["data_config"]

splits = yaml.safe_load(open(os.path.join(data_config["in_dir"], 'config.yaml'), 'r'))  
indices = splits.get(f'{dataset_name}_indices', [])    
transforms = get_transforms(dvs_res=data_config["dvs_res"])
dataset = ABCDataset(config=data_config, indexes=indices, transforms=transforms, ttc_mode="xy")
loader = DataLoader(dataset, num_workers=0, batch_size=1, pin_memory=True, shuffle=False)

# de normalize
task_name = config["data_config"]["outputs_list"][0] 
norm_target_fn =  copy.deepcopy(transforms)[task_name].transforms[0].to_device(device)
tensor_to_numpy_fn = TensorToNumpy()

# Initialize plot
def set_gray_grid(ax, title, xlabel, ylabel, xlim, ylim, fontsize, font): 
    ax.set_xlabel(xlabel, fontsize=fontsize, fontdict={'weight': 'bold'})
    ax.set_ylabel(ylabel, fontsize=fontsize, fontdict={'weight': 'bold'}) 
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')
    ax.tick_params(axis='both', labelsize=fontsize)
    
# Initialize plot
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(4, 4))
fontsize_axis=20
font = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'bold', 'size': fontsize_axis}
set_gray_grid(ax, "Move/Stay Prediction per Recording", "Time [ms]", "Time to Collision [ms]", (-150, 150), (-20, 250), fontsize_axis, font)

recording_lines = defaultdict(lambda: ([], [], [], []))  # (true_time, predicted_t, color, actual_t)
# Define color map for predictions
color_map = {True: "mediumseagreen", False: "crimson"}

# Initialize counters for direction accuracy
correct_t = 0
total_t = 0 

for step, data_dict in enumerate(loader):
    recording_id, recording_index, original_id = data_dict["origin"] 
    true_time = recording_index * 20  # ms
    
    input_dict = move_tensor_dict_to_device(data_dict["inputs"], device)
    target_dict = move_tensor_dict_to_device(data_dict["targets"], device)

    # Model prediction
    outputs = model(**input_dict)
    output_vector = tensor_to_numpy_fn(norm_target_fn.revert(outputs))[0]
    target_vector = tensor_to_numpy_fn(norm_target_fn.revert(target_dict[task_name]))[0]

    # Extract time to collision
    predicted_t = output_vector[2] 
    actual_t = target_vector[2]
    
    # Determine movement directions
    predicted_t_dir = predicted_t < 100 
    actual_t_dir = actual_t < 100 
    
    # Compare predictions with ground truth
    color = "silver"
    if predicted_t_dir == actual_t_dir:
        color = "mediumseagreen"
        correct_t += 1
    
    total_t += 1
    
    # Store data for line plot
    recording_lines[recording_id.item()][0].append(true_time)
    recording_lines[recording_id.item()][1].append(predicted_t)
    recording_lines[recording_id.item()][2].append(color)
    recording_lines[recording_id.item()][3].append(actual_t)

# Plot each recording line segment by segment
rec1, rec2, = None, None
for recording_id, (times, predictions, colors, targets) in recording_lines.items():
    smoothed_predictions = median_filter(predictions, size=window_size)  

    for i in range(1, len(times)):
        # Use smooth color transitions instead of binary colors 
        label = "" 
        if (rec1==None and colors[i]=="mediumseagreen"):
            label = "Correct"   
            rec1=1
        elif (rec2==None and colors[i]=="silver"):
            label = "Wrong" 
            rec2=1  
        plt.plot(times[i-1:i+1], targets[i-1:i+1], color=colors[i], linewidth=2, alpha=0.75, label=label) 

# Add legend and adjust grid
plt.tight_layout() 
plt.savefig(os.path.join(base_path, f"time_to_collision_plot_{model_name}_{dataset_name}.pdf"))

# Compute direction accuracy
accuracy_t = correct_t / total_t if total_t > 0 else 0.0
print(f"Accuracy (Move/Stay): {accuracy_t:.2f}")

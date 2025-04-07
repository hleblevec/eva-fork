import pdb, os, torch, yaml, copy
from torch.utils.data import DataLoader
from collections import defaultdict, deque
import numpy as np
import math
from dotenv import load_dotenv
from scipy.ndimage import median_filter, gaussian_filter1d
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as patches

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
from eva.data.dataset import GLCDataset
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

# override for real tests 
splits = yaml.safe_load(open(os.path.join(data_config["in_dir"], 'config.yaml'), 'r'))  
indices = splits.get(f'{dataset_name}_indices', [])    
transforms = get_transforms(dvs_res=data_config["dvs_res"])
dataset = GLCDataset(config=data_config, indexes=indices, transforms=transforms, ttc_mode="xy")
loader = DataLoader(dataset, num_workers=0, batch_size=1, pin_memory=True, shuffle=False)

# de normalize
task_name = config["data_config"]["outputs_list"][0] 
norm_target_fn =  copy.deepcopy(transforms)[task_name].transforms[0].to_device(device)
tensor_to_numpy_fn = TensorToNumpy()

# Initialize plot
def set_gray_grid(ax, title, xlabel, ylabel, xlim, ylim, fontsize, font): 
    ax.set_xlabel(xlabel, fontsize=fontsize, fontdict={'weight': 'bold'})
    ax.set_ylabel(ylabel, fontsize=fontsize, fontdict={'weight': 'bold'})
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')
    ax.tick_params(axis='both', labelsize=fontsize)
    
# Initialize plot
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(4, 4))

fontsize_axis=20
font = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'bold', 'size': fontsize_axis}
set_gray_grid(ax, "Side View", "-Z [cm]", "Y [cm]", (-20, 250), (-75, 150), fontsize_axis, font)

recording_lines = defaultdict(lambda: ([], [], [], []))
color_map = {True: "mediumseagreen", False: "crimson"}

for step, data_dict in enumerate(loader):
    recording_id, recording_index, original_id = data_dict["origin"] 
    
    input_dict = move_tensor_dict_to_device(data_dict["inputs"], device)
    target_dict = move_tensor_dict_to_device(data_dict["targets"], device)

    true_distance = data_dict["targets"]["distance"].item()
    x_ball, y_ball, z_ball = data_dict["targets"]["t_ball_pos"][0] / 10
    x_drone, y_drone, z_drone = data_dict["targets"]["t_drone_pos"][0] / 10

    # Model prediction
    outputs = model(**input_dict)
    output_vector = tensor_to_numpy_fn(norm_target_fn.revert(outputs))[0]
    target_vector = tensor_to_numpy_fn(norm_target_fn.revert(target_dict[task_name]))[0] 

    # Extract time to collision
    predicted_y = output_vector[1] 
    actual_y = target_vector[1]
    
    # Determine movement directions
    predicted_y_dir = np.sign(predicted_y)  # +1 (right), -1 (left), 0 (no movement) 
    actual_y_dir = np.sign(actual_y) 

    # Compare predictions with ground truth
    color = "silver"
    if predicted_y_dir == actual_y_dir:
        color = "mediumseagreen" 
    
    # Store data for line plot
    recording_lines[recording_id.item()][0].append(- x_ball.cpu() + x_drone.cpu())
    recording_lines[recording_id.item()][1].append(y_ball.cpu() - y_drone.cpu())
    recording_lines[recording_id.item()][2].append(z_ball.cpu() - z_drone.cpu())
    recording_lines[recording_id.item()][3].append(color)

# Plot each recording line segment by segment
rec1, rec2, = None, None
for recording_id, (x_ball, y_ball, z_ball, colors) in recording_lines.items(): 
    for i in range(1, len(x_ball)):
        # Use smooth color transitions instead of binary colors 
        label = "" 
        if (rec1==None and colors[i]=="mediumseagreen"):
            label = "Correct"   
            rec1=1
        elif (rec2==None and colors[i]=="silver"):
            label = "Wrong" 
            rec2=1 
        ax.plot(x_ball[i-1:i+1], z_ball[i-1:i+1], color=colors[i], linewidth=3, alpha=0.75, label=label) 


img_top = plt.imread("./docs/drone_icons/drone_icon_side.png")
imagebox_top = OffsetImage(img_top, zoom=0.15)
ab_top = AnnotationBbox(imagebox_top, (0, 0), frameon=False, xycoords='data', boxcoords="offset points", zorder=10, label="Drone")
ax.add_artist(ab_top)  

# Add legend and adjust grid    
plt.tight_layout()
plt.savefig(os.path.join(base_path, f"side_place_of_collision_plot_{model_name}_{dataset_name}.pdf"))
 

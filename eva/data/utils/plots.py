import os, pdb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.rcParams['font.family'] = 'Times New Roman'

from .dataset import GLCDataset
from .transforms import Normalize3DPoints
from torchvision import transforms

data_config = {
    "in_dir": "/datasets/pbonazzi/sony-rap/glc_dataset/vicon_aggregate/",
    "num_workers": 1,
    "batch_size": 8,
    "event_dt_ms": 10,
    "event_windows": 20,
    "event_accumulation": "addition",
    "event_decay_constant": 0.9,
    "imu_windows": 1,
    "frequency": "rgb",
    "max_num_of_e": 500,
    "overwrite": False,
    "outputs_list": ["position", "velocity"],
    "inputs_list": ["rgb"]
}

save_dir = os.path.join(data_config["in_dir"], "figures")
range_indexes = list(range(1, 185))

# 16.07.2024 
# 13, 18, 22, 38, 52, 53, 56, 69, 125 ball gets over the fence

# ID : Explaination 
# 4, 8, 13, 17, 18, 22, 31, 38, 72, 73, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 104, 113, 125, 134, 141, 149, 170
# 168, 169 : Ball Has Not Been Throwed 
# 14, 15, 16, 54 : Data Not Found

to_remove = [4, 8, 13, 14, 15, 16, 17, 18, 22, 31, 38, 54, 72, 73, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 104, 113, 125, 134, 141, 149, 168, 169, 170]
true_indexes = [idx for idx in range_indexes if idx not in to_remove]

true_indexes = true_indexes[:len(true_indexes)//8]
# true_indexes = true_indexes[len(true_indexes)//4:len(true_indexes)//2]
# true_indexes = true_indexes[len(true_indexes)//2:-len(true_indexes)//4]
# true_indexes = true_indexes[-len(true_indexes)//4:]

my_transforms = {
    "t_position": transforms.Compose([]),
    "t_velocity": transforms.Compose([]),
    "t_ttc": transforms.Compose([]),
    "i_dvs": transforms.Compose([]),
    "i_rgb": transforms.Compose([])
}
dataset = GLCDataset(config=data_config, indexes=true_indexes, transforms=my_transforms)
num_recordings = dataset.n_of_indexes
print("Data Initialized")

# Create a colormap with a distinct color for each recording
cmap = plt.get_cmap('rainbow', num_recordings)  # Use 'tab20' colormap with 20 colors (repeats if more than 20)
fig = plt.figure(figsize=(20, 6))

# Create a GridSpec with 2 rows and 3 columns
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

# Add subplots
ax1 = fig.add_subplot(gs[:, 0])  # First row, first column
ax2 = fig.add_subplot(gs[0, 1])  # First row, second column
ax3 = fig.add_subplot(gs[1, 1])  # Second row, second column
ax4 = fig.add_subplot(gs[:, 2], projection='3d')  # Span both rows, third column

# Increase font size individually for each subplot
for ax in [ax1, ax2, ax3]:
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(12)
    ax.title.set_fontsize(18)

# Define views
views = ['Top (XY-plane)', 'Side (XZ and YZ-plane)', ' ', '3D View (XYZ-plane)']
projections = [(0, 1), (0, 2), (1, 2), (0, 1, 2)]
axis_names = {0:"X", 1:"Y", 2:"Z"} 

origins_list = []

# Load the Data
for i in tqdm(range(len(dataset))):
    recording = dataset[i] 

    origin_index = recording['origin'][0] 
    color = cmap(origin_index)

    drone_pos = recording['targets']['t_drone_pos']
    ball_pos = recording['targets']['t_ball_pos']

    for ax, view, proj in zip([ax1, ax2, ax3, ax4], views, projections):
        if len(proj) == 2:
            i, j = proj
            scatter_drone = ax.scatter(drone_pos[i], drone_pos[j], color='black', marker='x')
            scatter_ball = ax.scatter(ball_pos[i], ball_pos[j], color=color, alpha=0.4)
        else:
            i, j, k = proj
            scatter_drone = ax.scatter(drone_pos[i], drone_pos[j], drone_pos[k], color='black', marker='x')
            scatter_ball = ax.scatter(ball_pos[i], ball_pos[j], ball_pos[k], color=color, alpha=0.4)

    if origin_index not in origins_list: 
        origins_list.append(origin_index)
        legend_origin = ax4.scatter([], [], [], color=color, marker='x', label=f'N_{true_indexes[origin_index]}') 
print("Data Plotted")

# Legends and Titles 
legend_drone = ax4.scatter([], [], [], color='black', marker='x', label='Drone (x)')
legend_ball = ax4.scatter([], [], [], color='black', label='Ball (o)')
ax4.legend(loc='upper right')
for ax, view, proj in zip([ax1, ax2, ax3, ax4], views, projections):
    ax.set_title(view)
    ax.set_xlabel(f'Axis {axis_names[proj[0]]}')
    ax.set_ylabel(f'Axis {axis_names[proj[1]]}')

    if axis_names[proj[0]] == "X" and axis_names[proj[1]] == "Y":
        ax.grid(True, alpha=0.6)
        ax.set_xticks(np.arange(-2500, 2500, 1000))
        ax.set_yticks(np.arange(-2500, 2500, 1000))
        ax.set_xlim(-2500, 2500)        
        ax.set_ylim(-2500, 2500)

    if axis_names[proj[1]] == "Z":
        ax.set_yticks(np.arange(0, 2000, 500))
        ax.set_ylim(0, 2000)
        ax.set_xticks(np.arange(-2500, 2500, 1000))
        ax.set_xlim(-2500, 2500)        

    if len(proj) == 3:
        ax.set_zlabel(f'Axis {axis_names[proj[2]]}')
        ax.set_zticks(np.arange(0, 2000, 500))
        ax.set_zlim(0, 2000)

    ax.set_aspect('equal')

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0, hspace=0)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'views.png'))

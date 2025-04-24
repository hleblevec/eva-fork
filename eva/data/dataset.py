import pdb
import bisect 
import os
import math
from itertools import accumulate
from tqdm import tqdm
import numpy as np
from scipy.stats import gaussian_kde
import yaml
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision import transforms


from .subclasses import GLCFlight  
from .transforms import Normalize3DPoints

class ABCDataset(Dataset):
    def __init__(self, indexes, config, transforms, ttc_mode="xy"): 

        self.inputs_list = config["inputs_list"]        
        self.outputs_list = config["outputs_list"]     

        self.dataset = {}
        self.dataset_indexes = {} 
        self.n_of_indexes = len(indexes) 
        self.ttc_mode = ttc_mode 

        count = []
        with ThreadPoolExecutor(max_workers=config["num_workers"]) as executor:
            futures = {executor.submit(self._load_flight, i, value, config): i for i, value in enumerate(indexes)}
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    i = futures[future]
                    self.dataset[i], length = future.result() 
                    self.dataset_indexes[i] = length 
                except Exception as exc:
                    count.append(i)
                    print(f"Error with recording # {i}")   
                
        original_keys = list(self.dataset.keys()) 
        key_mapping = {old_key: new_key for new_key, old_key in enumerate(original_keys)} 
        self.dataset = {key_mapping[old_key]: value for old_key, value in self.dataset.items()}

        # Accumulate the indexes in order
        sorted_indexes = sorted(self.dataset_indexes.keys())
        ordered_lengths = [self.dataset_indexes[i] for i in sorted_indexes]
        self.list_idx_boundaries = np.asarray(list(accumulate(ordered_lengths)))

        self.resolution = torch.float32
        self.transforms = transforms

    def _load_flight(self, i, value, config):
        flight = GLCFlight(id=value, config=config, ttc_mode=self.ttc_mode) 
        return flight, len(flight)
        
    def __len__(self):
        return self.list_idx_boundaries[-1]

    def _match_recording_index(self, index):
        recording_id = bisect.bisect_right(self.list_idx_boundaries, index)
        recording_index = index if recording_id == 0 else (index - self.list_idx_boundaries[recording_id - 1])
        return recording_id, recording_index

    def __getitem__(self, index):
        recording_id, recording_index = self._match_recording_index(index)
        item = self.dataset[recording_id][recording_index] 
        input_dict, target_dict, vis_dict = {}, {}, {}

        # raw values
        target_dict['t_drone_pos'] = torch.as_tensor(item['drone']['pos'], dtype=self.resolution) 
        target_dict['t_ball_pos'] = torch.as_tensor(item['ball']['pos'], dtype=self.resolution) 
        target_dict['t_ball_time'] = torch.as_tensor(item['ball']['t'], dtype=self.resolution) 
        target_dict['t_collision_pos'] = torch.as_tensor(item['collision'], dtype=self.resolution)  
        target_dict['t_time_to_collision'] = torch.as_tensor(item['time_to_collision'], dtype=self.resolution)  

        x1, y1, z1 = item['ball']['pos']
        x2, y2, z2 = item['drone']['pos'] 
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        target_dict['distance'] = torch.as_tensor(distance, dtype=self.resolution) 

        # Target 
        if "delta_ball_2_drone_3d_xyz" in self.outputs_list:  
            delta_ball_2_drone_3d_xyz = torch.as_tensor(item['ball']['pos'] - item['drone']['pos'], dtype=self.resolution) 
            target_dict['delta_ball_2_drone_3d_xyz'] = self.transforms["delta_ball_2_drone_3d_xyz"](delta_ball_2_drone_3d_xyz)  

        if "delta_ball_2_point_of_collision_yzt" in self.outputs_list: 
            yz = (target_dict['t_ball_pos'] - target_dict['t_collision_pos'])[-2:]
            yzt = torch.cat([yz, target_dict['t_time_to_collision'].unsqueeze(0)]) 
            target_dict['delta_ball_2_point_of_collision_yzt'] = self.transforms["delta_ball_2_point_of_collision_yzt"](yzt)

        if "delta_drone_2_point_of_collision_yzt" in self.outputs_list:  
            yz = (target_dict['t_drone_pos'] - target_dict['t_collision_pos'])[-2:]
            yzt = torch.cat([yz, target_dict['t_time_to_collision'].unsqueeze(0)]) 
            target_dict['delta_drone_2_point_of_collision_yzt'] = self.transforms["delta_drone_2_point_of_collision_yzt"](yzt)

        # Input 
        if "dvs" in self.inputs_list:
            t, c, h, w = item['dvs']['data'].shape
            i_dvs = torch.as_tensor(item['dvs']['data'], dtype=self.resolution).view(t * c, h, w) 
            input_dict['i_dvs'] = self.transforms["i_dvs"](i_dvs)   
            vis_dict['i_dvs_raw'] = self.transforms["i_dvs"].transforms[0](i_dvs)
        
        if "imu" in self.inputs_list:
            input_dict['i_imu_g'] = torch.as_tensor(item['imu']['g'], dtype=self.resolution)
            input_dict['i_imu_a'] = torch.as_tensor(item['imu']['a'], dtype=self.resolution)
        
        if "rgb" in self.inputs_list:  
            t, c, h, w = item['rgb']['data'].shape 
            i_rgb = torch.as_tensor(item['rgb']['data'], dtype=self.resolution).reshape(t * c, h, w) 
            input_dict['i_rgb'] =  self.transforms["i_rgb"](i_rgb)   

        return {"inputs": input_dict, "targets": target_dict, "vis_dict": vis_dict,  "origin": (recording_id, recording_index, item["original_id"])}


# Define a function to standardize grid settings
def set_gray_grid(ax, title, xlabel, ylabel, xlim, ylim, fontsize, font):
    ax.set_title(title, fontsize=fontsize+5, fontdict={'weight': 'bold'})
    ax.set_xlabel(xlabel, fontsize=fontsize, fontdict={'weight': 'bold'})
    ax.set_ylabel(ylabel, fontsize=fontsize, fontdict={'weight': 'bold'})
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')
    ax.tick_params(axis='both', labelsize=fontsize)

def plot_3d_positions_over_time(dataset, save_path):

    splits = yaml.safe_load(open(os.path.join(data_config["in_dir"], 'config.yaml'), 'r'))
    train_indices = splits.get(f'train_indices', [])
    val_indices = splits.get(f'val_indices', [])
    test_indices = splits.get(f'test_indices', [])
    test_dpu_indices = splits.get(f'test_dpu_indices', [])

    # Styles
    train_color = "#0072B2" 
    val_color = "#E69F00" 
    test_color = "#009E73"
    test_dpu_color = "#CC79A7"
    train_line = '-' 
    val_line = '-.' 
    test_line = '--' 
    test_dpu_line = ':'
    
    fontsize_axis=20
    font = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'bold', 'size': fontsize_axis}


    # Create a figure with 1 row and 4 columns
    fig, axs = plt.subplots(1, 4, figsize=(20, 5)) 
    axs = axs.flatten() 
    ax_3d = fig.add_subplot(1, 4, 1, projection='3d') 


    # Loop through the dataset and extract the delta_ball_2_drone__3d_xyz positions
    for i in tqdm(range(len(dataset))):
        # Extract the 3D position
        x_pos, y_pos, z_pos = dataset[i]["targets"]["delta_ball_2_drone_3d_xyz"]  

        # Determine the color and marker based on the recording_id
        if dataset[i]["origin"][-1] in train_indices:
            color = train_color 
            line_type = train_line 
        elif dataset[i]["origin"][-1] in val_indices:
            color = val_color 
            line_type = val_line
        elif dataset[i]["origin"][-1] in test_indices:
            line_type = test_line
            color = test_color 
        else:
            line_type = test_dpu_line 
            color = test_dpu_color 

        # Store the positions for each recording (for drawing lines)
        if i == 0 or dataset[i]["origin"][-1] != dataset[i-1]["origin"][-1]: 
            x_rec, y_rec, z_rec = [], [], []

        # Append the current position to the list of the current recording
        x_rec.append(x_pos / 10)
        y_rec.append(y_pos / 10)
        z_rec.append(z_pos / 10)

        # If this is the last entry of the recording (last frame), plot the line and final point
        if i == len(dataset) - 1 or dataset[i]["origin"][-1] != dataset[i+1]["origin"][-1]: 
            x_rec_inverted = [-x for x in x_rec]
            
            axs[1].plot(y_rec, x_rec_inverted, color=color, linestyle=line_type, linewidth=1, alpha=0.5)
            axs[2].plot(x_rec_inverted, z_rec, color=color, linestyle=line_type, linewidth=1, alpha=0.5)
            axs[3].plot(y_rec, z_rec, color=color, linestyle=line_type, linewidth=1, alpha=0.5)
            ax_3d.plot(x_rec, y_rec, z_rec, color=color, linestyle=line_type, linewidth=1, alpha=0.5)

            # Mark the last point with a larger, distinct marker
            axs[1].scatter(y_rec[-1], -x_rec[-1], color=color, s=50, edgecolors='black', zorder=5, marker="o")
            axs[2].scatter(-x_rec[-1], z_rec[-1], color=color, s=50, edgecolors='black', zorder=5, marker="o")
            axs[3].scatter(y_rec[-1], z_rec[-1], color=color, s=50, edgecolors='black', zorder=5, marker="o")
            ax_3d.scatter(x_rec[-1], y_rec[-1], z_rec[-1], color=color, s=50, edgecolors='black', zorder=5, marker='o')

    # Set titles and labels outside the loop
    set_gray_grid(axs[1], "Top View", "X [cm]", "-Z [cm]", (-150, 150), (-20, 250), fontsize_axis, font)
    set_gray_grid(axs[2], "Side View", "-Z [cm]", "Y [cm]", (-20, 250), (-75, 150), fontsize_axis, font)
    set_gray_grid(axs[3], "Back View", "X [cm]", "Y [cm]", (-150, 150), (-75, 150), fontsize_axis, font)

    axs[0].set_axis_off()
    ax_3d.set_title("3D View", fontsize=fontsize_axis+5, fontdict={'weight': 'bold'})
    ax_3d.set_xlabel("Z [cm]", fontsize=fontsize_axis, labelpad=fontsize_axis, fontdict={'weight': 'bold'})
    ax_3d.set_ylabel("X [cm]", fontsize=fontsize_axis, labelpad=fontsize_axis, fontdict={'weight': 'bold'})
    ax_3d.set_zlabel("", fontsize=fontsize_axis, labelpad=fontsize_axis, fontdict={'weight': 'bold'})
    ax_3d.text(-100, 200, 120, "Y [cm]",  fontsize=fontsize_axis, color="black", ha='center', va='center', zorder=10, fontdict={'weight': 'bold'})
    ax_3d.grid(True, linestyle='--', linewidth=0.7, color='black')

    # Make the 3D axis ticks integers
    ax_3d.set_zlim(-75, 150) 
    ax_3d.set_xticks(np.linspace(-250, 20, num=3))  
    ax_3d.set_yticks(np.linspace(-150, 150, num=3))  
    ax_3d.set_zticks(np.linspace(-75, 150, num=3))  

    # Drone Images for use as a marker
    img_top = plt.imread("./docs/drone_icons/drone_icon_top.png")
    img_side = plt.imread("./docs/drone_icons/drone_icon_side.png")
    img_front = plt.imread("./docs/drone_icons/drone_icon_front.png")
    imagebox_top = OffsetImage(img_top, zoom=0.05)
    ab_top = AnnotationBbox(imagebox_top, (0, 0), frameon=False, xycoords='data', boxcoords="offset points", zorder=10)
    axs[1].add_artist(ab_top)  
    imagebox_side = OffsetImage(img_side, zoom=0.15)
    ab_side = AnnotationBbox(imagebox_side, (0, 0), frameon=False, xycoords='data', boxcoords="offset points", zorder=10)
    axs[2].add_artist(ab_side)  
    imagebox_front = OffsetImage(img_front, zoom=0.15)
    ab_front = AnnotationBbox(imagebox_front, (0, 0), frameon=False, xycoords='data', boxcoords="offset points", zorder=10)
    axs[3].add_artist(ab_front) 
    #axs[2].annotate('', xy=(-1000, -500), xytext=(0, 0), arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->', lw=2)) 
    #axs[2].text(-1000, -500,  "Plexiglass",  fontsize=18, color="red", backgroundcolor=(1, 1, 1, 1), ha='center', va='center', zorder=10)

    # Increase tick label size for all axes
    for ax in axs:
        ax.tick_params(axis='both', labelsize=18)
    ax_3d.tick_params(axis='both', labelsize=14)  

    # Dummy scatter plots to create legend handles
    ax_3d.scatter([], [], [], color=train_color, label="Train.", s=100, edgecolors='black')
    ax_3d.scatter([], [], [], color=val_color, label="Valid.", s=100, edgecolors='black')
    ax_3d.scatter([], [], [], color=test_color, label="Test 1", s=100, edgecolors='black')
    ax_3d.scatter([], [], [], color=test_dpu_color, label="Test 2", s=100, edgecolors='black')
    train_handle = Line2D([0], [0], color=train_color, linestyle=train_line, lw=2, label="Train.")
    val_handle = Line2D([0], [0], color=val_color, lw=2,  linestyle=val_line, label="Valid.")
    test_handle = Line2D([0], [0], color=test_color, lw=2,  linestyle=test_line,label="Test 1")
    test_dpu_handle = Line2D([0], [0], color=test_dpu_color, lw=2,  linestyle=test_dpu_line, label="Test 2")

    ax_3d.legend(
        handles=[train_handle, val_handle, test_handle, test_dpu_handle],
        loc='upper left',
        fontsize=16,   
        frameon=True,
        fancybox=True,
        framealpha=0.7, 
        edgecolor='black', 
        title="Datasets",  
        title_fontsize=18, 
        handlelength=1.5, 
        handleheight=1.5,  
        ncol=1 
    )

    # Adjust layout for better spacing between plots 
    plt.tight_layout()
    plt.savefig(save_path, dpi=300) 

def compute_min_max(dataset, outputs_var, save_path): 
    all_out_var = []
    count = 0 
    for i in tqdm(range(len(dataset))):   
        all_out_var.append(dataset[i]["targets"][outputs_var])  
    # # get min/max
    norm_fn = Normalize3DPoints() 
    norm_fn.fit(torch.stack(all_out_var)) 

    print("Min : ", norm_fn.min_vals)
    print("Max : ", norm_fn.max_vals) 

def draw_delta_drone_2_point_of_collision_yzt(dataset, save_path): 
    all_x = []
    all_y = []
    all_t = []
    all_id = []
    original_ids = []

    # Loop through the dataset to extract x and y
    for i in tqdm(range(len(dataset))): 
        delta_drone = dataset[i]["targets"]["delta_drone_2_point_of_collision_yzt"]  
        all_id.append(dataset[i]["origin"][0])
        original_ids.append(dataset[i]["origin"][2])
        all_x.append(delta_drone[0].item())
        all_y.append(delta_drone[1].item())
        all_t.append(delta_drone[2].item())

    # Rectangle parameters
    rect_width = 600
    rect_height = 600
    center_x, center_y = 0, 0

    # Identify points inside the rectangle
    np.random.seed(42) 
    inside_mask = (np.abs(np.array(all_x) - center_x) <= rect_width / 2) & (np.abs(np.array(all_y) - center_y) <= rect_height / 2)
    points_inside_x = np.array(all_x)[inside_mask]
    points_inside_y = np.array(all_y)[inside_mask]

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.scatter(all_x, all_y, color='lightgray', label='Outside Points')
    plt.scatter(points_inside_x, points_inside_y, color='red', label='Inside Rectangle')
    plt.axvline(x=center_x - rect_width / 2, color='grey', linestyle='--')  # left border
    plt.axvline(x=center_x + rect_width / 2, color='grey', linestyle='--')  # right border
    plt.axhline(y=center_y - rect_height / 2, color='grey', linestyle='--')  # bottom border
    plt.axhline(y=center_y + rect_height / 2, color='grey', linestyle='--')  # top border
    plt.title('Points Inside and Outside the Rectangle', fontsize=16)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.legend(loc="upper right", fontsize=14)
    plt.grid(True)
    plt.savefig(save_path) 

    inside_indices = np.where(inside_mask)[0]
    outside_indices = np.where(~inside_mask)[0]
    num_inside = len(inside_indices)
    inside_recording_ids = np.array(original_ids)[inside_indices]
    outside_recording_ids = np.array(original_ids)[outside_indices] 

    unique_inside_recordings = np.unique(inside_recording_ids)
    unique_outside_recordings = np.unique(outside_recording_ids) 
 
    n_inside = len(unique_inside_recordings)
    n_outside = len(unique_outside_recordings) 
    print(n_inside, n_outside)
    print("n_inside, n_outside")
    train_size = int(0.7 * n_inside)
    val_size = int(0.2 * n_inside)
    test_size = n_inside - (train_size + val_size) 
    np.random.shuffle(unique_inside_recordings)
    np.random.shuffle(unique_outside_recordings) 

    train_inside = unique_inside_recordings[:train_size]
    val_inside = unique_inside_recordings[train_size:train_size + val_size]
    test_inside = unique_inside_recordings[train_size + val_size:] 
    
    train_outside = unique_outside_recordings[:train_size]
    val_outside = unique_outside_recordings[train_size:train_size + val_size]
    test_outside = unique_outside_recordings[train_size + val_size:] 
    
    train_indices = np.concatenate([train_inside, train_outside])
    val_indices = np.concatenate([val_inside, val_outside])
    test_indices = np.concatenate([test_inside, test_outside])

    config = {'train_indices': train_indices.tolist(), 'val_indices': val_indices.tolist(), 'test_indices': test_indices.tolist()}
    with open(os.path.join(data_config["in_dir"], 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
 

def draw_delta_drone_2_point_of_collision_yzt_colors(dataset, save_path, label): 
    all_x = []
    all_y = []
    all_t = []

    # Loop through the dataset to extract x and y
    for i in tqdm(range(len(dataset))): 
        delta_drone = dataset[i]["targets"]["delta_drone_2_point_of_collision_yzt"] 
        all_x.append(delta_drone[0].item())
        all_y.append(delta_drone[1].item())
        all_t.append(delta_drone[2].item())

    plt.figure(figsize=(10, 8), dpi=300)

    # Define color scheme for each group
    color_map = {"train": 'blue', "val": 'green', "test": 'red', "outlier": 'orange'}  

    # Scatter plot with jittered data points 
    plt.scatter(all_x, all_y, color=color_map[label], s=30, alpha=0.6, label=label)   

    # Formatting the plot
    plt.title(f'Scatter Plot of Collision Points {label}', fontsize=18, pad=20)
    plt.xlabel('X Coordinate (mm)', fontsize=18, labelpad=10)
    plt.ylabel('Y Coordinate (mm)', fontsize=18, labelpad=10)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12) 
    plt.xlim(-600, 600) 
    plt.ylim(-600, 600) 
    plt.yticks(fontsize=12) 
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_stats_for_paper(dataset):
    import numpy as np
    from collections import defaultdict

    # Dictionaries to store data for each recording (based on original_id)
    elevations = defaultdict(list)
    lengths = defaultdict(list)
    speeds = defaultdict(list)

    # Dictionary to track the last entry for each recording
    previous_entry = {}

    # Iterate through the dataset
    for i in range(len(dataset)):
        target = dataset[i]["targets"]
        x, y, z = target["t_ball_pos"]  # Unpack the current ball position
        ball_time = target["t_ball_time"]  # Current ball time
        _, _, original_id = dataset[i]["origin"]  # Recording ID

        # Append elevation (z-coordinate) to the corresponding recording
        elevations[original_id].append(z)  # Z-coordinate

        # Calculate length (distance from origin in 3D space) and append
        length = np.linalg.norm([x, y, z])  # Euclidean distance from origin
        lengths[original_id].append(length)

        # Check if a previous entry exists for this recording
        if original_id in previous_entry:
            prev_pos, prev_time = previous_entry[original_id]
            prev_z, prev_y, prev_x = prev_pos

            # Calculate distance traveled between consecutive points
            distance = np.linalg.norm([x - prev_x, y - prev_y, z - prev_z])  # 3D Euclidean distance
            time_diff = ball_time - prev_time  # Time elapsed

            # Ensure valid time difference to calculate speed
            if time_diff > 0:
                speed = distance / time_diff
                speeds[original_id].append(speed)

        # Update the previous entry for the current recording
        previous_entry[original_id] = ((z, y, x), ball_time)

    # Calculate statistics for each metric
    def calculate_stats(data_dict):
        """Calculate Min, Max, Avg, SD for a dictionary of lists."""
        all_values = []
        for key, values in data_dict.items():
            all_values.extend(values)
        return {
            "Min": np.min(all_values) if all_values else None,
            "Max": np.max(all_values) if all_values else None,
            "Avg": np.mean(all_values) if all_values else None,
            "SD": np.std(all_values) if all_values else None,
        }

    elevation_stats = calculate_stats(elevations)
    length_stats = calculate_stats(lengths)
    speed_stats = calculate_stats(speeds)

    # Print the results for table insertion
    print("Elevation (m):", elevation_stats)
    print("Length (m):", length_stats)
    print("Speed (m/s):", speed_stats)



def compute_collision_locations(dataset):
    splits = yaml.safe_load(open(os.path.join(data_config["in_dir"], 'config.yaml'), 'r'))
    train_indices = splits.get(f'train_indices', [])
    val_indices = splits.get(f'val_indices', [])
    test_indices = splits.get(f'test_indices', [])
    test_dpu_indices = splits.get(f'test_dpu_indices', [])
    
    data_dicts = {
        "train": {"top-right": 0, "top-left": 0, "down-right": 0, "down-left": 0},
        "val": {"top-right": 0, "top-left": 0, "down-right": 0, "down-left": 0},
        "test": {"top-right": 0, "top-left": 0, "down-right": 0, "down-left": 0},
        "test_dpu": {"top-right": 0, "top-left": 0, "down-right": 0, "down-left": 0},
    } 
    processed_origins = set()
    for i in tqdm(range(len(dataset))): 
        x_pos, y_pos, z_pos = dataset[i]["targets"]["delta_drone_2_point_of_collision_yzt"]   
        if x_pos < 0:
            if y_pos < 0:
                collision_location = "down-left"
            else:
                collision_location = "top-left"
        else:
            if y_pos < 0:
                collision_location = "down-right"
            else:
                collision_location = "top-right"
 
        origin = dataset[i]["origin"][-1] 
        if origin in processed_origins:
            continue 
        processed_origins.add(origin)
        
        if origin in train_indices:
            split = "train"
        elif origin in val_indices:
            split = "val"
        elif origin in test_indices:
            split = "test"
        elif origin in test_dpu_indices:
            split = "test_dpu"
        else:
            continue 
        data_dicts[split][collision_location] += 1
        
    print(data_dicts)

if __name__ == '__main__': 
 
    data_config = {
        "in_dir": os.getenv("ABCD_DATA_PATH"),
        "num_workers": 10,
        "batch_size": 8,
        "event_dt_ms": 20,
        "rgb_windows": 1,
        "event_windows": 1,
        "event_polarities": 1,
        "event_polarities_mode": "substract",
        "event_accumulation": "addition",
        "event_decay_constant": 0.1,
        "imu_windows": 1,
        "frequency": "dvs",
        "max_num_of_e": 500,
        "overwrite": False,
        "outputs_list": ["delta_drone_2_point_of_collision_yzt"],
        "inputs_list": ["dvs"]
    }  

    splits = yaml.safe_load(open(os.path.join(data_config["in_dir"], 'config.yaml'), 'r'))  
    train_indices = splits.get(f'train_indices', [])    
    val_indices = splits.get(f'val_indices', [])    
    test_indices = splits.get(f'test_indices', [])    
    test_dpu_indices = splits.get(f'test_dpu_indices', [])      
    
    true_indexes = train_indices + val_indices + test_indices + test_dpu_indices   
    my_transforms = {"delta_ball_2_point_of_collision_yzt":  transforms.Compose([]), 
                     "delta_drone_2_point_of_collision_yzt":  transforms.Compose([]), 
                     "delta_ball_2_drone_3d_xyz":  transforms.Compose([]), 
                     "i_dvs": transforms.Compose([transforms.Resize(size=(80, 80), interpolation=InterpolationMode.NEAREST) ]), 
                     "i_rgb": transforms.Compose([])} 
    save_dir=os.path.join(data_config["in_dir"], "figures")
    
    dataset = ABCDataset(config=data_config, indexes=true_indexes, transforms=my_transforms) 
        

    calculate_stats_for_paper(dataset) 

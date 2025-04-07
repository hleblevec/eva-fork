import pdb
import yaml
import torch  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchprofile

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Function to calculate FLOPs
def calculate_flops(model, input_size):
    model.eval()
    with torchprofile.profile(model, inputs=(torch.randn(input_size),)) as prof:
        flops = prof.total("flops")
    return flops

def is_pickleable(obj):
    import pickle
    try:
        pickle.dumps(obj)
        return True
    except Exception as e:
        print(f"Object {obj} is not pickleable: {e}")
        return False

def read_csv_pandas(filename, sep=",", quotechar='"', dtype=None, chunksize=None): 
    return pd.read_csv(filename, sep=sep, quotechar=quotechar, dtype=dtype, chunksize=chunksize, engine='c')

def write_pandas_csv(df, filename):
    return df.to_csv(filename)

def save_list_to_txt(lst, filename):
    with open(filename, 'w', encoding='utf-8') as f: 
        f.write('\n'.join(map(str, lst)))

def load_list_from_txt(filename):
    with open(filename, 'r', encoding='utf-8') as f: 
        return [line.strip() for line in f]

def load_yaml_from_txt(filename): 
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config


def plot_occupancy_grid(distances):
    from ..transforms import VICON_DATASET_AGGREGATE
    min_vals, max_vals = VICON_DATASET_AGGREGATE["t_position"]
    min_vals, max_vals = torch.tensor(min_vals), torch.tensor(max_vals) 

    # Create an empty 3D grid   
    size = (max_vals - min_vals+ torch.tensor([1,1,1])).int().numpy()
    occupancy_grid = np.zeros((size.astype(int)))

    # Populate the occupancy grid based on the drone positions
    for pos in distances:
        norm_xyz = (pos - min_vals).int() 
        x, y, z = norm_xyz
        occupancy_grid[x, y, z] = 1

    # Plotting the 3D occupancy grid
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = occupancy_grid.nonzero()
    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig('3d_occupancy_plot.png')
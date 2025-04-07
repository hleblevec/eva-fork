import os
import pdb
import numpy as np
import pandas as pd
import math
import pdb
from ..utils.utils import read_csv_pandas

from scipy.spatial.transform import Rotation as R

def transform_points(points, camera_pos, camera_rot):
    # Create rotation matrix from quaternion
    camera_rot_matrix = R.from_quat(camera_rot).as_matrix()
    # Invert the rotation and translation to transform from world to camera coordinates
    inv_rot_matrix = camera_rot_matrix.T
    inv_translation = -inv_rot_matrix @ camera_pos

    # Transform points to camera coordinate system
    points_cam = (inv_rot_matrix @ (points - camera_pos).T).T
    return points_cam

def project_points(points, focal_length=1.0):
    # Assuming a pinhole camera model with focal length along the z-axis
    points_proj = points / points[:, 2, np.newaxis]  # Normalize by z to project onto image plane
    points_2d = points_proj[:, :2] * focal_length
    return points_2d


def fill_first_non_zero_vectorized(arr):
  """
  Fills the first rows of a NumPy array with the first non-zero values using vectorized operations.

  Args:
      arr: A NumPy array.

  Returns:
      A new NumPy array with the first rows filled with the first non-zero values.
  """
  # Find the first non-zero row index
  first_non_zero_idx = np.where(~np.all(arr == 0, axis=1))[0][0]

  # Get the first non-zero row
  first_non_zero_row = arr[first_non_zero_idx].copy()

  # Create a mask for zero rows
  zero_mask = np.all(arr == 0, axis=1)

  # Fill zero rows with the copied non-zero row
  filled_arr = arr.copy()
  filled_arr[zero_mask] = first_non_zero_row

  return filled_arr 


def read_vicon_csv(path):
    if not os.path.exists(path):  
        print(f"No Pose File Found : {path}") 
        return {} 
    
    file = read_csv_pandas(path, quotechar='"', sep=';').iloc[4:] 
    header_key = file.keys().to_list()[0]
    file_data = file[header_key].str.split(',', expand=True) 
    column_names = header_key.split(',')
    column_dict = {column_names[i] : i for i in range(len(column_names))}
    
    # Replace empty strings
    file_data.replace('', pd.NA, inplace=True)   
    file_data = file_data.apply(pd.to_numeric, errors='coerce')

    # Linear Interpolate and Fill NaN 
    file_data.interpolate(method='linear', inplace=True) 
    file_data.fillna(0, inplace=True)  
    file_data = file_data.to_numpy()  
    
    num = file_data.shape[0]  
    b_r = file_data[:, column_dict['ball_RX']:column_dict['ball_RW']+1].astype(float)
    b_p = file_data[:, column_dict['ball_TX[mm]']:column_dict['ball_TZ[mm]']+1].astype(float) 
    b_p = fill_first_non_zero_vectorized(b_p) 

    d_r = file_data[:, column_dict['drone_RX']:column_dict['drone_RW']+1].astype(float)
    d_p = file_data[:, column_dict['drone_TX[mm]']:column_dict['drone_TZ[mm]']+1].astype(float) 
    d_p = fill_first_non_zero_vectorized(d_p) 
    
    vicon_time = file_data[:, column_dict['VICONTime[s]']].astype(float) 
    fpga_time = file_data[:, column_dict['FPGATime[s]']].astype(float)   
 
    # Assuming all ball points have the same camera transformation
    camera_pos = d_p[0]
    camera_rot = d_r[0]

    # Transform points to camera space and project to 2D
    #points_cam = transform_points(b_p, camera_pos, camera_rot)
    #ball_2d = project_points(points_cam)

    return {"vicon_time": vicon_time, "fpga_time": fpga_time, "ball": {"pos": b_p, "rot": b_r}, "drone": {"pos": d_p, "rot": d_r}}






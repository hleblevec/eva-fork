import torch
import pdb 
import numpy as np

VICON_DATASET_AGGREGATE = {
    "delta_drone_2_point_of_collision_yzt": [[-459.5156, -553.5490,   -7.1910], [593.4048, 581.6352, 550.1083]],
    #"delta_drone_2_point_of_collision_yzt": [[-782.7958, -1389.8550, -7.1910], [1053.5048,  581.6352,  550.1083]],
    "delta_ball_2_point_of_collision_yzt": [[-1255.4928,  -334.8084,  -257.6307], [1097.6951, 1353.4250,  450.0846]],
    "delta_ball_2_drone_3d_xyz": [[-3493.6396, -1842.1871,  -596.6235], [-129.5871, 1448.3530, 1377.0629]], 
}


class Normalize3DPoints:
    def __init__(self, key=None):
        if key is None:
            self.min_vals = None
            self.max_vals = None
        else:
            self.load(key)

        self.eps = 2**-6

    def load(self, key): 
        self.min_vals = torch.tensor(VICON_DATASET_AGGREGATE[key][0])
        self.max_vals = torch.tensor(VICON_DATASET_AGGREGATE[key][1])

    def to_device(self, device): 
        self.min_vals = self.min_vals.to(device)
        self.max_vals = self.max_vals.to(device)
        return self

    def fit(self, points):
        # Calculate the min and max values for each coordinate (x, y, z)
        self.min_vals = torch.min(points, dim=0).values
        self.max_vals = torch.max(points, dim=0).values

    def __call__(self, points):
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Normalization parameters are not initialized. Call 'fit' with data first.")

        # Normalize to range [-1, 1 - delta]
        normalized_points = (2 - self.eps) * (points - self.min_vals) / (self.max_vals - self.min_vals) - 1
        return normalized_points

    def revert(self, normalized_points):
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Normalization parameters are not initialized. Call 'fit' or 'load' with data first.")
    
        # Revert normalization to original range
        original_points = ((normalized_points + 1) / (2 - self.eps)) * (self.max_vals - self.min_vals) + self.min_vals
        return original_points


class TensorToNumpy:
    def __init__(self): 
        pass
    def __call__(self, data):   
        data = data.cpu()
        data = data.detach()
        data = data.numpy()
        return data

class CorrectWhiteBalance:
    def __init__(self): 
        pass
    def __call__(self, img):   
        img = img.astype(np.float32)
        
        # Compute the average of each channel
        avg_r = np.mean(img[:, :, 2])  # Red channel
        avg_g = np.mean(img[:, :, 1])  # Green channel
        avg_b = np.mean(img[:, :, 0])  # Blue channel
        
        # Scale each channel based on the gray world assumption
        avg_gray = (avg_r + avg_g + avg_b) / 3
        scale_r = avg_gray / avg_r
        scale_g = avg_gray / avg_g
        scale_b = avg_gray / avg_b
        
        # Apply scaling to each channel
        img[:, :, 2] *= scale_r
        img[:, :, 1] *= scale_g
        img[:, :, 0] *= scale_b
        
        # Clip values to valid range [0, 255]
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

class NormalizeRangeRobust:
    def __init__(self, low_perc=5, high_perc=95, eps=2**-7):
        self.low_perc = low_perc
        self.high_perc = high_perc
        self.eps = eps

    def __call__(self, data):
        # Compute percentiles
        low_val = torch.quantile(data, self.low_perc / 100.0)
        high_val = torch.quantile(data, self.high_perc / 100.0)

        # Avoid division by zero
        if high_val == low_val:
            return torch.zeros_like(data).to(data.device)

        # Clip outliers
        data = torch.clamp(data, min=low_val, max=high_val)

        # Normalize to range [-1, 1 - eps]
        normalized_data = (2 - self.eps) * (data - low_val) / (high_val - low_val) - 1
        return normalized_data
    
class NormalizeRange:
    def __init__(self):
        self.eps = 2**-7

    def __call__(self, data):  
        # Calculate min and max
        data_min = data.min()
        data_max = data.max()

        # Avoid division by zero in case data_max equals data_min
        if data_max == data_min:
            return torch.zeros_like(data).to(data.device) 

        # Normalize data to range [-1, 1 - self.eps]
        normalized_data = (2 - self.eps) * (data - data_min) / (data_max - data_min) - 1
        return normalized_data

class NormalizeImage:
    def __init__(self):
        pass
    
    def revert(self, data):
        return data*255.0

    def __call__(self, data): 
        data = data.float() 
        return data / 255.0

class NormalizeTTC:
    def __init__(self):
        pass
    
    def revert(self, data):
        return data*4.0

    def __call__(self, data): 
        data = data.float() 
        return data/4.0
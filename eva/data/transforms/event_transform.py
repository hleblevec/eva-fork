import pdb
import math
import numpy as np


class SpatialBinning:
    def __init__(self, new_h, new_w):
        self.new_h, self.new_w = new_h, new_w
        
    def __call__(self, img):
        """
        Resize an image by summing pixels in non-overlapping blocks.
        
        img: PyTorch tensor of shape (C, H, W) 
        """
        C, H, W = img.shape
        stride_h, stride_w = H // self.new_h, W // self.new_w

        # Ensure the image size is divisible by the new size
        assert H % self.new_h == 0 and W % self.new_w == 0, "New size must be a divisor of original size"
        
        img = img.unfold(1, stride_h, stride_h).unfold(2, stride_w, stride_w)  # Shape: (C, new_h, new_w, stride_h, stride_w)
        img = img.sum(dim=(-1, -2))  # Sum over the small patches
        return img


class EventRandomDrop:
    def __init__(self, drop_probability=0.1):
        """
        Initialize the EventRandomDrop class.

        Args:
            drop_probability (float): The probability of dropping an event (0 to 1).
        """
        self.drop_probability = drop_probability

    def __call__(self, data):
        """
        Randomly drop events in the data array.

        Args:
            data (np.ndarray): The input data with shape (c, t, w, h), where
                               c is the number of channels,
                               t is the temporal dimension,
                               w is the width, and
                               h is the height.

        Returns:
            np.ndarray: The data array with random events dropped.
        """
        # Generate a random mask with the same shape as the input data
        mask = np.random.rand(*data.shape) > self.drop_probability
        
        # Apply the mask to the data (set dropped events to 0)
        data = data * mask
        
        return data.float()
    
class EventUniformNoise:
    def __init__(self, noise_level=0.1):
        """
        Initialize the AddUniformNoise class.

        Args:
            noise_level (float): The maximum amplitude of the uniform noise to add.
        """
        self.noise_level = noise_level

    def __call__(self, data):
        """
        Add uniform noise to the data array.

        Args:
            data (np.ndarray): The input data with shape (c, t, w, h).

        Returns:
            np.ndarray: The data array with added uniform noise.
        """
        # Generate uniform noise with the same shape as the input data
        noise = np.random.uniform(-self.noise_level, self.noise_level, size=data.shape)
        
        # Add the noise to the data
        data = data + noise
        
        return data.float()
    
class EventReduceVoxel:
    def __init__(self):
        pass
    def __call__(self, data):
        if data.max() > 1: 
            if data.shape[1] == 2:
                data[:, 0, ...][ data[:, 1, ...] >= data[:, 0, ...]] = 0  # if ch 1 has more evs than 0
                data[:, 1, ...][ data[:, 1, ...] < data[:, 0, ...]] = 0  # if ch 0 has more evs than 1
                data = data.clip(0,1) 
            
        return data

class EventAccumulation:
    def __init__(self, num_bins, decay_constant, polarities, polarities_mode, mode="addition"):
        self.num_bins = num_bins 
        self.decay_constant = decay_constant
        self.polarities = polarities 
        self.polarities_mode = polarities_mode
        
        assert mode in ["addition", "fractional"]
        self.mode = mode 


    def _addition(self, data, values):
        x, y, p, t = values 

        if self.num_bins==1: 
            if self.polarities==1:
                if self.polarities_mode == "substract":
                    np.add.at(data[0, 0], (y[p == 0], x[p == 0]), -1)
                    np.add.at(data[0, 0], (y[p == 1], x[p == 1]), 1)
                    data.clip(-128, 127)
                elif self.polarities_mode == "merge":
                    np.add.at(data[0, 0], (y, x), 1)    
                    data.clip(0, 255)
            else:
                np.add.at(data[0, 0], (y[p == 0], x[p == 0]), 1)
                np.add.at(data[0, 1], (y[p == 1], x[p == 1]), 1)
                data.clip(0, 255)

            return {"data": data, "timestamps": np.array([t[-1]]), "t_start": t[0], "t_end": t[-1]}
            
        fixed_ts = np.linspace(t[0], t[-1], self.num_bins)
        for i, f_ts in enumerate(fixed_ts): 
            if i ==0:
                v_ids = (t<=f_ts)
            else:
                v_ids = (t<=f_ts) & (t>fixed_ts[i-1])

            if self.polarities==1: 
                if self.polarities_mode == "substract":
                    np.add.at(data[i, 0], (y[v_ids][p[v_ids] == 0], x[v_ids][p[v_ids] == 0]), -1)
                    np.add.at(data[i, 0], (y[v_ids][p[v_ids] == 1], x[v_ids][p[v_ids] == 1]), 1)
                    data.clip(-128, 127)
                elif self.polarities_mode == "merge":
                    np.add.at(data[i, 0], (y[v_ids], x[v_ids]), 1) 
                    data.clip(0, 255)
            else:
                np.add.at(data[i, 0], (y[v_ids][p[v_ids] == 0], x[v_ids][p[v_ids] == 0]), 1)
                np.add.at(data[i, 1], (y[v_ids][p[v_ids] == 1], x[v_ids][p[v_ids] == 1]), 1)
                data.clip(0, 255)

        return {"data": data, "timestamps": fixed_ts, "t_start": t[0], "t_end": t[-1]}

    def _interpolate_decay(self, event_timestamp, given_timestamp, decay_constant):
        time_difference = given_timestamp - event_timestamp
        interpolated_value = math.exp(-decay_constant * time_difference)
        return interpolated_value

    def _fractional(self, data, values):

        raise NotImplementedError
    
        # x, y, p, t = values 
            

        # fixed_ts = np.linspace(t[0], t[-1], self.num_bins) 
        # for i, f_ts in enumerate(fixed_ts):
        #     # Find events that fall into the current bin
        #     if i ==0:
        #         v_ids = (t<=f_ts)
        #     else:
        #         v_ids = (t<=f_ts) & (t>fixed_ts[i-1])
        #     # Calculate fractional contributions for events spanning multiple bins 
        #     true_indices = np.where(v_ids)[0]
        #     for j in true_indices:  
        #         # Add events
        #         contribution = self.interpolate_decay(t[j], f_ts, decay_constant=self.decay_constant) 
        #         if self.polarities==1:
        #             np.add.at(data[i, 0], (y[v_ids], x[v_ids]), contribution)
        #         else:
        #             np.add.at(data[i, 0], (y[v_ids][p[v_ids] == 0], x[v_ids][p[v_ids] == 0]), contribution)
        #             np.add.at(data[i, 1], (y[v_ids][p[v_ids] == 1], x[v_ids][p[v_ids] == 1]), contribution)

        # return {"data": data, "timestamps": fixed_ts, "t_start": t[0], "t_end": t[-1]}
    
    def __call__(self, data, values):

        x = np.array(values["x"])
        y = np.array(values["y"])
        p = np.array(values["p"])  
        t = np.array(values["t"])   


        if self.mode =="addition": 
            data_dict = self._addition(data, (x, y, p, t))

        elif self.mode =="fractional":
            data_dict = self._fractional(data, (x, y, p, t))

        return data_dict


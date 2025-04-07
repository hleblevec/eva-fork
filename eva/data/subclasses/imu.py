import os
import pdb
import ast
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from ..utils.utils import save_list_to_txt, load_list_from_txt

ACC_SCALING = 2*8.0/2**16 #g
GYRO_SCALING = 2*2000.0/2**16 #dps

class IMU:
    def __init__(self, path=None, verbose=False, overwrite=True, vis_mode=False):
        # Initialized Variables
        self.current_idx = 0
        self.gyro_history, self.acc_history = [], []
        self.timestamps=[]
        self.values=[]

        if vis_mode:
            return 

        # Init Paths 
        assert os.path.exists(path), f"FileNotFound Error {path}"
        self.timestamps_path = path.replace(".RAW", "_time.txt")
        self.values_path = path.replace(".RAW", "_values.txt")
        if overwrite:
            if os.path.exists(self.timestamps_path): os.remove(self.timestamps_path)
            if os.path.exists(self.values_path): os.remove(self.values_path)

        # Initialize File 
        self._indexfile(path) 

        if verbose:
            print(f"Indexed IMU : {path}")


    def _indexfile(self, path): 

        if not os.path.exists(self.timestamps_path) or not os.path.exists(self.values_path):  
            raw_file = open(path, mode="rb")
            raw_file.seek(0,2)
            num_bytes = raw_file.tell()  
            raw_file.seek(0,0)
            lenght = num_bytes // (8+6+6)

            for index in tqdm(range(0, lenght)):
                i = (8+6+6) * index 
                raw_file.seek(i,0)

                # timestamps
                ts_str = raw_file.read(2)
                ts_lsb1 = int.from_bytes(ts_str, byteorder='little')
                ts_str = raw_file.read(2)
                ts_lsb2 = int.from_bytes(ts_str, byteorder='little')
                ts_str = raw_file.read(2)
                ts_msb1 = int.from_bytes(ts_str, byteorder='little')
                ts_str = raw_file.read(2)
                ts_msb2 = int.from_bytes(ts_str, byteorder='little')
                timestamp  = round(float(((ts_msb1 << 48) + (ts_msb2 << 32) + (ts_lsb1<<16) + (ts_lsb2<<0)) * 10e-9), 8)
                self.timestamps.append(timestamp) 

                # gyroscope
                ts_str = raw_file.read(2)
                IMU_gx = int.from_bytes(ts_str, byteorder='little', signed=True)
                ts_str = raw_file.read(2)
                IMU_gy = int.from_bytes(ts_str, byteorder='little', signed=True)
                ts_str = raw_file.read(2)
                IMU_gz = int.from_bytes(ts_str, byteorder='little', signed=True)
                
                # acceleration
                ts_str = raw_file.read(2)
                IMU_ax = int.from_bytes(ts_str, byteorder='little', signed=True)
                ts_str = raw_file.read(2)
                IMU_ay = int.from_bytes(ts_str, byteorder='little', signed=True)
                ts_str = raw_file.read(2)
                IMU_az = int.from_bytes(ts_str, byteorder='little', signed=True)
                
                gyroscope = [IMU_gx*GYRO_SCALING, IMU_gy*GYRO_SCALING, IMU_gz*GYRO_SCALING]
                accelerometer = [IMU_ax*ACC_SCALING, IMU_ay*ACC_SCALING, IMU_az*ACC_SCALING]
                self.values.append([gyroscope, accelerometer]) 

            raw_file.close() 
            save_list_to_txt(self.timestamps, self.timestamps_path)  
            save_list_to_txt(self.values, self.values_path)  
            self.timestamps = np.asarray(self.timestamps)
            self.values = np.asarray(self.values)
        else:
            self.timestamps = np.asarray(load_list_from_txt(self.timestamps_path)).astype(float)  
            self.values = np.array([ast.literal_eval(item) for item in np.asarray(load_list_from_txt(self.values_path))], dtype=float)

    def __len__(self):
        return len(self.values)

    def cut_recording(self, index_array):
        self.values = self.values[index_array]
        self.timestamps = self.timestamps[index_array]

    def __repr__(self):
        description = "IMU(samples="+str(self.__len__())+")"
        return description

    def get_closest(self, t_start, t_end):   
        # WARNING : Loading only the last one  
        up_bound = self.timestamps - t_start 
        low_bound = self.timestamps - t_end
        idx = np.where((low_bound <= 0)&(up_bound >= 0))[0]
        if len(idx) == 0:
            return {'t_start': 0.0, 't_end': 0.0, 'g': [0.0,0.0,0.0], 'a': [0.0,0.0,0.0]}
        else:
            return self.__getitem__(np.max(idx))

    def __getitem__(self, index):
        self.current_idx = index 
        
        t_end  = self.timestamps[index]
        t_start = t_end if index==0 else self.timestamps[index-1]

        # gyroscope, accellerometer
        gyroscope = self.values[index][0]
        accelerometer = self.values[index][1]
        
        return {"t_start": t_start, "t_end": t_end, "g" : gyroscope, "a": accelerometer}

    def plot(self, ax, item, save_path=None, auto_limits=False): 
        """Enhanced 3D plotting function for accelerometer and gyroscope data."""
        
        # Update history
        self.acc_history.append(item["a"]) 
        self.gyro_history.append(item["g"])  

        # Clear the plot
        ax.clear() 

        # Plot accelerometer and gyroscope data points
        ax.scatter(
            item["a"][0], item["a"][1], item["a"][2], 
            c='red', marker='o', s=100, label='Accelerometer', edgecolors='black', alpha=0.9
        ) 
        ax.scatter(
            item["g"][0], item["g"][1], item["g"][2], 
            c='blue', marker='^', s=100, label='Gyroscope', edgecolors='black', alpha=0.9
        ) 

        # Plot the trajectories for the last 5 points
        if len(self.acc_history) > 1:
            acc_xs, acc_ys, acc_zs = zip(*self.acc_history[-5:])
            ax.plot(acc_xs, acc_ys, acc_zs, color='salmon', linewidth=2, alpha=0.8, label="")

        if len(self.gyro_history) > 1:
            gyro_xs, gyro_ys, gyro_zs = zip(*self.gyro_history[-5:])
            ax.plot(gyro_xs, gyro_ys, gyro_zs, color='lightblue', linewidth=2, alpha=0.8, label="")

        # Set axes labels and limits
        ax.set_xlabel('X', fontsize=18, labelpad=10, alpha=0.8)
        ax.set_ylabel('Y', fontsize=18, labelpad=10, alpha=0.8)
        ax.set_zlabel('Z', fontsize=18, labelpad=0, alpha=0.8)

        if auto_limits:
            # Adjust limits dynamically
            all_data = np.array(self.acc_history[-5:] + self.gyro_history[-5:])
            x_min, x_max = all_data[:, 0].min() - 1, all_data[:, 0].max() + 1
            y_min, y_max = all_data[:, 1].min() - 1, all_data[:, 1].max() + 1
            z_min, z_max = all_data[:, 2].min() - 1, all_data[:, 2].max() + 1
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
        else:
            # Static limits
            ax.set_xlim(-7, 7)
            ax.set_ylim(-7, 7)
            ax.set_zlim(-7, 7)

        # Fewer ticks for cleaner visualization
        pad = 1
        ax.set_xticks(np.linspace(ax.get_xlim()[0]+pad, ax.get_xlim()[1]-pad, 5))
        ax.set_yticks(np.linspace(ax.get_ylim()[0]+pad, ax.get_ylim()[1]-pad, 5)) 
        z_ticks = np.linspace(ax.get_zlim()[0]+pad, ax.get_zlim()[1]-pad, 5) 
        ax.set_zticks(z_ticks) 
        middle_tick = z_ticks[2]
        ax.set_zticklabels(['' if tick == middle_tick else f'{tick:.0f}' for tick in z_ticks])  

        # Make sure the ticks are not too close to each other (avoid overlap)
        ax.tick_params(axis='both', which='major', labelsize=18, length=5, width=1, direction='inout', grid_color='gray', grid_alpha=0.5)

        # Add legend
        ax.legend(loc='upper right', fontsize=20, markerscale=2)

        #ax.set_title(item['t_end'])   
            
if __name__ == '__main__': 

    base_path = "/datasets/pbonazzi/sony-rap/glc_dataset/2_vicon_dummy_dataset2/"
    rec = IMU(path=os.path.join(base_path, f"IMU2.RAW"), verbose=False, overwrite=False) 

    from ..utils import is_pickleable
    is_pickleable(rec)

    pdb.set_trace()
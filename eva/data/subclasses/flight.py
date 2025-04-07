import os
import pdb
import fire
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm
from scipy.optimize import minimize_scalar, brentq 

from .rgb import RGB
from .imu import IMU
from .dvs import DVS
from .object import Object
from .vicon import read_vicon_csv


class GLCFlight:
    def __init__(self, id, config, ttc_mode): 

        # Recording 
        in_dir = os.path.join(config["in_dir"], "data")
        out_dir = os.path.join(config["in_dir"], "figures")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "recordings"), exist_ok=True)
        self.save_file = os.path.join(out_dir, "recordings", str(id)+".mp4")
        self.original_id = id
 
        # Sensors 
        self.frequency = config["frequency"] 
        self.inputs_list =  config["inputs_list"] 
        
        self.data_list = self.inputs_list.copy()
        if self.frequency not in self.data_list:
            self.data_list.append(self.frequency) 

        self.imu = IMU(path=os.path.join(in_dir, "imu", f"IMU{id}.RAW"), verbose=False, overwrite=config["overwrite"]) 
        self.rgb = RGB(path=os.path.join(in_dir, "rgb",f"RGB{id}.RAW"), verbose=False, overwrite=config["overwrite"], rgb_windows=config["rgb_windows"])
        self.dvs = DVS(path=os.path.join(in_dir, "dvs",f"DVS{id}.RAW"), imu_ts=self.imu.timestamps, verbose=False, overwrite=config["overwrite"], config=config)  

        # Objects
        vicon_data = read_vicon_csv(path=os.path.join(in_dir,  "vicon", f"VICON_IMU{id}.csv"))    
        
        # Ball
        self.ball = Object(name="Ball", c_point_traj=['r', 'salmon'])   
        self.ball.set_data(pos=vicon_data["ball"]["pos"], rot=vicon_data["ball"]["rot"], time=vicon_data["fpga_time"])

        # Drone
        self.drone = Object(name="Drone", c_point_traj=['b', 'lightblue'])  
        self.drone.set_data(pos=vicon_data["drone"]["pos"], rot=vicon_data["drone"]["rot"], time=vicon_data["fpga_time"]) 
        
        # cut physical barrier
        self.ttc_mode = ttc_mode
        self._cut_recording_x(buffer=300) 
        
        index = self._compute_index_of_collision()
        self.actual_pos = self.ball.positions[index]
        self.actual_ttc = self.ball.timestamps[index] 
         
        #self.actual_ttc = self.ttc_minimize_scalar() # - 0.1
        #self.actual_pos = self.ball.interpolate_pos_vel(self.actual_ttc)["pos"]
        self._cut_recording_after(self.actual_ttc) 
        
        # cut after time of collisions  
        #if len(self.ball) < 55 : raise ValueError 
        self._cut_recording_before(self.ball.timestamps[-45]) 

    def __len__(self):
        return len(self.__dict__[self.frequency])

    def __getitem__(self, index, asynchronous=True):  
        # Get synchronized timestamps
        data_dict, ts = self._get_sensors_items(index, asynchronous) 

        # Interpolates position of object
        data_dict["ball"] = self.ball.interpolate_pos_vel(ts)  
        data_dict["drone"] = self.drone.interpolate_pos_vel(ts) 

        data_dict["time_to_collision"] = (self.actual_ttc - ts)*1000 # ms 
        data_dict["collision"] = self.actual_pos
        data_dict["original_id"] = self.original_id 

        return data_dict

    def _compute_index_of_collision(self):  
        distances = np.linalg.norm(self.drone.positions[:, :2] - self.ball.positions[:, :2], axis=1) 
        return np.argmin(distances)

    def _cut_recording_after(self, ts):         
        self.drone.cut_recording(index_array=self.drone.timestamps <= ts)
        self.ball.cut_recording(index_array=self.ball.timestamps <= ts)  
        self.imu.cut_recording(index_array=self.imu.timestamps <= ts)
        self.rgb.cut_recording(index_array=self.rgb.timestamps <= ts)   
        self.dvs.cut_recording(index_array=self.dvs.events["t"] <= ts) 

    def _cut_recording_before(self, ts):    
        self.drone.cut_recording(index_array=self.drone.timestamps >= ts)
        self.ball.cut_recording(index_array=self.ball.timestamps >= ts)  
        self.imu.cut_recording(index_array=self.imu.timestamps >= ts)
        self.rgb.cut_recording(index_array=self.rgb.timestamps >= ts)   
        self.dvs.cut_recording(index_array=self.dvs.events["t"] >= ts) 
        
    def _cut_recording_distances(self, buffer):    
        distances = np.linalg.norm(self.ball.positions[:, :2] - self.drone.positions[:, :2], axis=1)
        remove_indices = np.nonzero(distances < buffer)[0] 
        if remove_indices.size > 0:   
            cut_time = self.ball.timestamps[remove_indices[0]]
            self._cut_recording_after(cut_time)  

    def _cut_recording_x(self, buffer):     
        distances = self.drone.positions[:, 0] - self.ball.positions[:, 0]
        remove_indices = np.nonzero(distances < buffer)[0] 
        if remove_indices.size > 0:   
            cut_time = self.ball.timestamps[remove_indices[0]]   
            self._cut_recording_after(cut_time.item())   

    def _get_sensors_items(self, index, asynchronous):

        # Loading single events doesnt make too much sense
        if self.frequency == "dvs": 
            main = self.__dict__[self.frequency].slice_with_dt(index)
        else:
            try:
                main = self.__dict__[self.frequency][index]
            except:
                pdb.set_trace()
                
        data_dict = {}
        for key in self.data_list :
            if key == self.frequency:
                data_dict[key] = main
            elif key == "rgb" and asynchronous==True: 
                data_dict["rgb"] = self.rgb.get_next_async(main["t_start"], main["t_end"])
            else: 
                data_dict[key] = self.__dict__[key].get_closest(main["t_start"], main["t_end"])
        return data_dict, main["t_end"]

    def ttc_minimize_scalar(self):  
        # Calculate Time Boundaries
        min_ts0, max_ts0 = self.drone.timestamps[0], self.drone.timestamps[-1]
        min_ts1, max_ts1 = self.ball.timestamps[0], self.ball.timestamps[-1] 
        lower_bound, upper_bound = max(min_ts0, min_ts1), min(max_ts0, max_ts1)

        # Find the scalar that minimize the distance between the two  
        time_of_collision = minimize_scalar(self._compute_distance_drone2ball_at_tmst, bounds=(lower_bound, upper_bound), method='bounded').x
        return time_of_collision

    def _compute_distance_drone2ball_at_tmst(self, timestamp): 
        pos_drone = self.drone.interpolate_pos_vel(timestamp)["pos"]
        pos_ball = self.ball.interpolate_pos_vel(timestamp)["pos"]
        if self.ttc_mode == "xy": return np.linalg.norm(pos_drone[:2] - pos_ball[:2]) 
        elif self.ttc_mode == "xyz": return np.linalg.norm(pos_drone - pos_ball)

    def plot(self, save_file=None):
        # Initialize Plot
        if save_file is None:
            save_file = self.save_file 
             
        # Combine both pieces of information in a single title
        first_dict = self.__getitem__(0) 

        # Create subfigures with better spacing 
        fig, axs = plt.subplots(1, 3, figsize=(15, 5)) 
        axs = axs.flatten() 
        ax_3d = fig.add_subplot(1, 3, 3, projection='3d')  
        axs[2].set_axis_off() 

        # Define additional plot settings for clarity 
        axes_map = {'rgb': axs[0], 'dvs': axs[1], 'ball': ax_3d, 'drone': ax_3d} 
        indexes = {"rgb": None, "dvs": None, "ball": None, "drone": None} 

        # Update Function
        def update(frame):
            data_dict = self.__getitem__(frame)
            save_image = frame in [1, len(self)//2, len(self)-1]
            for key, ax in axes_map.items():
                if key == "dvs": 
                    self.__dict__[key].plot(ax, data_dict[key], transpose=True)
                elif key == "rgb":
                    self.__dict__[key].plot(ax, data_dict[key], resize=True)
                else:
                    self.__dict__[key].plot(ax, data_dict[key]) 

            if save_image:
                save_path = f"output/{frame}.pdf"  
                plt.savefig(save_path)

        # Create Video
        frames = len(self) 
        with tqdm(total=frames) as pbar:
            def update_with_progress(frame):
                update(frame)
                pbar.update(1)
            ani = FuncAnimation(fig, update_with_progress, frames=frames, blit=False, interval=1)
            ani.save(save_file, writer='ffmpeg', fps=30)



def main(id=None):

    # Base Configs
    base_config = { 
        "outputs_list": ["delta_drone_2_point_of_collision_yzt"],
        "inputs_list": ["imu", "dvs", "rgb"]
    }

    # Initialize DataModule
    data_config = {
        "in_dir": os.getenv("ABCD_DATA_PATH"),
        "num_workers": 10,
        "batch_size": 8,
        "imu_windows": 1,
        "rgb_windows": 1,
        "event_windows": 1,
        "event_dt_ms": 10,
        "event_polarities": 2,
        "event_polarities_mode": "substract",
        "event_accumulation": "addition",
        "event_decay_constant": 0.1,
        "frequency": "rgb", 
        "overwrite": False,
        "outputs_list": base_config["outputs_list"],
        "inputs_list": base_config["inputs_list"]
    }
    
    # Single Flight Visualization
    if id is not None:
        print("Flight", id)
        flight = GLCFlight(id=id, config=data_config, ttc_mode="xy") 
        flight.plot()
    else:
        # Multi Flight Visualization
        for i in tqdm(range(270, 301)):
            flight = GLCFlight(id=i, config=data_config, ttc_mode="xy") 
            flight.plot() 

if __name__ == "__main__":
    fire.Fire(main)

import os
import pdb
import numpy as np
import cv2
import matplotlib.pyplot as plt

from ..utils.utils import save_list_to_txt, load_list_from_txt, read_csv_pandas

class Object:
    def __init__(self, name, c_point_traj=['r', 'salmon'], overwrite=True,verbose=False): 

        # Initialized Variables
        self.current_idx = None
        self.name = name
        self.position_history = []
        self.c_point_traj = c_point_traj

        # Data
        self.positions = []
        self.timestamps = []
        self.rotations = []

    def _create_vicon_dummy(self, ts_range, trajectory="random", n=100):
        self.timestamps = np.linspace(ts_range[0], ts_range[1], n) # ms
        self.rotations = np.random.rand(n, 3)

        if trajectory=="random":
            self.positions = np.random.rand(n, 3)
        elif trajectory=="parabolic":
            self._create_parabolic(ts_range)
        elif trajectory=="zeros":
            self.positions = np.zeros((n, 3)) 

    def _create_parabolic(self, ts_range, start=[0.5,0.5,0.5], end=[0,0,0]):
        def x(t, x_start, x_end=0):
            return x_start + (x_end-x_start) * t

        def y(t, y_start, y_end=0):
            return y_start + (y_end-y_start) * t

        def z(t, z_start, z_end=0):
            return 4 * (z_start - z_end) * (t - 0.5)**2 + z_end

        t = (self.timestamps - ts_range[0]) / (ts_range[1] - ts_range[0])
        x_values = x(t, start[0], end[0])
        y_values = y(t, start[1], end[1])
        z_values = z(t, start[2], end[2]) 
        self.positions = np.column_stack([x_values, y_values, z_values])

    def _read_vicon_csv(self, path):
         # Frame, SubFrame, RX, RY, RZ, TX, TY, TZ
        file = read_csv_pandas(path, quotechar='"', sep=';').iloc[4:] 
        file_data = file["Objects"].str.split(',', expand=True).to_numpy() 
        self.timestamps = file_data[:, 1:2].astype(float)
        self.rotations = file_data[:, 2:5].astype(float)
        self.positions = file_data[:, 5:8].astype(float) 

    def set_data(self, time, pos, rot):
        self.timestamps = time
        self.positions = pos
        self.rotations = rot

    def cut_recording(self, index_array):  
        self.timestamps = self.timestamps[index_array]
        self.positions = self.positions[index_array.flatten(), :]
        self.rotations = self.rotations[index_array.flatten(), :]

    def add_positions(self, positions):   
        self.positions.append(positions) 

    def __len__(self):
        return len(self.timestamps)

    def __repr__(self):
        description = str(self.name)+"(idx="+str(self.current_idx)+")"
        return description

    def _find_closest_index(self, timestamp): 
            differences = self.timestamps - timestamp
            id_list = np.where(differences <= 0)[0] 
            if len(id_list) > 0:
                min_i = id_list[-1] 
                max_i = min(min_i+1, len(self.timestamps)-1) if min_i != 0 else min_i
                return min_i, max_i
            return 0, 0

    def interpolate_pos_vel(self, timestamp, count=True):
        min_i, max_i = self._find_closest_index(timestamp) 
        if count:
            self.current_idx = min_i

        if min_i == max_i:
            return {"pos": self.positions[min_i], "vel": [0,0,0], "t": timestamp}

        # Calculate interpolation weights
        t0 = self.timestamps[min_i]
        t1 = self.timestamps[max_i]
        w1 = (timestamp - t0) / (t1 - t0)
        w0 = 1.0 - w1
        
        # Interpolate the position
        inter_position = w0 * self.positions[min_i] + w1 * self.positions[max_i]

        # Calculate velocity
        inter_velocity = np.array((self.positions[max_i] - self.positions[min_i]) / (t1 - t0))

        return {"pos": inter_position, "vel": inter_velocity, "t": timestamp}

    def __getitem__(self):
        pass 

    def plot(self, ax, item, save_path=None, auto_limits=False):  
        """Enhanced plotting function with transparent text and fewer ticks."""
        
        # Update position history 
        fontsize_base = 25
        
        # Clear the plot for Ball (if required)
        if self.name == "Ball":
            ax.clear()  

        # Extract and round position coordinates
        p1, p2, p3 = map(lambda x: (round(x, 1))/10, item["pos"])  
        self.position_history.append((p1, p2, p3))  

        # Plot the current position
        ax.scatter(p1, p2, p3, c=self.c_point_traj[0], marker='o', s=100, label=self.name, edgecolors='black')  

        # Plot trajectory (last 5 points)
        if len(self.position_history) > 1:
            xs, ys, zs = zip(*self.position_history[-5:])
            ax.plot(xs, ys, zs, color=self.c_point_traj[1], linewidth=2, alpha=0.8)

        # Set axes labels, limits, and ticks for "Drone"
        if self.name == "Drone":
            ax.set_xlabel('Z [cm]', fontsize=fontsize_base, labelpad=fontsize_base, alpha=0.8, fontweight='bold')
            ax.set_ylabel('X [cm]', fontsize=fontsize_base, labelpad=fontsize_base, alpha=0.8, fontweight='bold')
            ax.text(100, 200, 120, "Y [cm]",  fontsize=fontsize_base, color="black", ha='center', va='center', zorder=10, fontdict={'weight': 'bold'}) 

            if auto_limits:
                # Automatically set limits based on position history
                x_min, x_max = min(xs) - 100, max(xs) + 100
                y_min, y_max = min(ys) - 100, max(ys) + 100
                z_min, z_max = min(zs) - 100, max(zs) + 100
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_zlim(z_min, z_max)
            else:
                # Static limits
                ax.set_xlim(-250, 150)
                ax.set_ylim(-150, 150)
                ax.set_zlim(-150, 150)  
            
            # Set ticks on each axis with improved parameters
            pad = 50
            ax.tick_params(axis='both', which='major', labelsize=fontsize_base-5, length=5, width=1, direction='inout', grid_color='gray', grid_alpha=0.5)

            ax.set_xticks(np.linspace(ax.get_xlim()[0] + pad, ax.get_xlim()[1] - pad, 3))
            ax.set_yticks(np.linspace(ax.get_ylim()[0] + pad, ax.get_ylim()[1] - pad, 3)) 
            z_ticks = np.linspace(ax.get_zlim()[0], ax.get_zlim()[1], 3) 
            ax.set_zticks(z_ticks) 
            middle_tick = z_ticks[2:4]
            ax.set_zticklabels(['' if tick in middle_tick else f'{tick:.0f}' for tick in z_ticks])  

            # Add legend with bold font
            ax.legend(loc='upper left', fontsize=fontsize_base, markerscale=2) 
            
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)  

if __name__ == '__main__': 
    ball = Object(name="Ball", c_point_traj=['r', 'salmon'])

    from ..utils import is_pickleable
    is_pickleable(ball)

    pdb.set_trace()
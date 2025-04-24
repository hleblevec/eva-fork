import os
import pdb
import numpy as np
import cv2

from eva.data.transforms import Normalize3DPoints 

class Position:
    def __init__(self, name, output_type, c_point_traj=['r', 'salmon'], master=True, overwrite=True, verbose=False): 

        # Initialized Variables
        self.current_idx = None
        self.name = name
        self.position_history = []
        self.c_point_traj = c_point_traj
        self.master = master

        # Data
        self.positions = []  
        self.position_history = [] 

        self.fn_norm = Normalize3DPoints(output_type)

    def add_positions(self, positions):    
        self.positions.append(positions)  

    def reset(self):
        del self.positions  
        del self.position_history
        self.positions = []  
        self.position_history = []

    def __len__(self):
        return len(np.vstack(self.positions))

    def __repr__(self):
        return str(self.name)+"(idx="+str(self.current_idx)+")" 

    def __getitem__(self, index): 
        return np.vstack(self.positions)[index]

    def plot(self, ax, current_pos):   

        if self.name == "targets":
            ax.clear()  

        # plot point
        p1, p2, p3 = round(current_pos[0], 0), round(current_pos[1],0), round(current_pos[2],0) 
        ax.scatter(p1, p2, p3, c=self.c_point_traj[0], marker='o', label=self.name) 
        ax.text(p1, p2, p3, str([p1,p2,p3]), size=8, color=self.c_point_traj[1])
       
        # plot history
        self.position_history.append([p1,p2,p3])
        if len(self.position_history) > 1:
            xs, ys, zs = zip(*self.position_history[-10:])
            ax.plot(xs, ys, zs, color=self.c_point_traj[1], linewidth=1)

        if self.name == "predictions":
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            ax.set_xlim(self.fn_norm.min_vals[0], self.fn_norm.max_vals[0])
            ax.set_ylim(self.fn_norm.min_vals[1], self.fn_norm.max_vals[1])
            ax.set_zlim(self.fn_norm.min_vals[2], self.fn_norm.max_vals[2])
            ax.legend(loc=0)

if __name__ == '__main__': 
    ball = Object(name="Ball", c_point_traj=['r', 'salmon'])

    from ..utils import is_pickleable
    is_pickleable(ball)

    pdb.set_trace()
import numpy as np
import torch
import os
import threading
import pdb
from tqdm import tqdm

import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

from .position import Position 

from msand.data.subclasses import RGB, IMU, DVS
from msand.data.transforms import Normalize3DPoints 

class Plotter:
    def __init__(self, config): 
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(211, projection='3d')

        # inputs
        self.imu = IMU(vis_mode=True) 
        self.rgb = RGB(vis_mode=True)
        self.dvs = DVS(vis_mode=True, config=config)  

        self.security_distance_mm = 200

        # outputs
        self.predictions = Position(name="predictions", output_type=config["outputs_list"][0],  c_point_traj=['r', 'salmon'])   
        self.targets = Position(name="targets", output_type=config["outputs_list"][0],  c_point_traj=['b', 'lightblue'])   
        self.reset() 

        self.is_direct = 'delta_drone_2_point_of_collision_yzt' == config["outputs_list"][0] 
        self.output_key = config["outputs_list"][0] 
        self.fn_norm = Normalize3DPoints(config["outputs_list"][0])  
        
    def update_stats(self, targets, outputs, batch, dataset_name):   
        # Inputs 
        if dataset_name != "train": 
            if "i_dvs" in list(batch["inputs"].keys()):
                self.dvs.add_data(batch["inputs"]["i_dvs"].detach().cpu()) 

        if self.is_direct: 
            self.update_stats_delta_yzt(targets, outputs, batch, dataset_name)
        else: 
            self.update_stats_delta_3d_xyz(targets, outputs) 

    def update_stats_delta_3d_xyz(self, targets, outputs):   

        # Standard 
        t_position = self.fn_norm.revert(targets[self.output_key].detach().cpu())
        o_position = self.fn_norm.revert(outputs[self.output_key].detach().cpu())

        # Compute distances and drifts
        pred_direction_errors = np.linalg.norm(t_position - o_position, axis=1) 

        # Update running statistics for distances
        self.distances_count += len(pred_direction_errors)
        self.distances_sum += np.sum(pred_direction_errors)
        self.distances_sum_sq += np.sum(pred_direction_errors**2) 

        # Determine predictions based on distance threshold and temporal consistency
        num_of_predictions = len(pred_direction_errors)
        predicted_triggers = np.zeros(num_of_predictions)
        target_triggers = np.zeros(num_of_predictions)
        for i in range(num_of_predictions):
            if (np.abs(o_position[i][0]) <= self.security_distance_mm) and (np.abs(o_position[i][1]) <= self.security_distance_mm):
                predicted_triggers[i] = 1
            if (np.abs(t_position[i][0]) <= self.security_distance_mm) and (np.abs(t_position[i][1]) <= self.security_distance_mm):
                target_triggers[i] = 1

        # Calculate precision, recall, F1 score, and confusion matrix  
        precision = precision_score(target_triggers, predicted_triggers, zero_division=0)
        recall = recall_score(target_triggers, predicted_triggers, zero_division=0)
        f1 = f1_score(target_triggers, predicted_triggers, zero_division=0)
        accuracy = accuracy_score(target_triggers, predicted_triggers)
        conf_matrix = confusion_matrix(target_triggers, predicted_triggers, labels=[True, False])
        tn, fp, fn, tp = conf_matrix.ravel()
        self.precision += precision
        self.recall += recall
        self.f1 += f1
        self.accuracy += accuracy
        self.conf_matrix += conf_matrix
        self.lenght_num += 1


    def update_stats_delta_yzt(self, targets, outputs, batch, dataset_name):

        # Standard 
        real_distance = targets["distance"].detach().cpu().numpy() 
        t_yzt = self.fn_norm.revert(targets[self.output_key].detach().cpu()).detach().cpu().numpy()
        o_yzt = self.fn_norm.revert(outputs[self.output_key].detach().cpu()).detach().cpu().numpy()

        # Update running statistics for distances 
        delta_yz = np.sqrt(np.sum((t_yzt[:, :2] - o_yzt[:, :2]) ** 2, axis=1)) 
        self.delta_yz_count += len(delta_yz)
        self.delta_yz_sum += np.sum(delta_yz)
        self.delta_yz_sum_sq += np.sum(delta_yz**2) 
        delta_t = np.sqrt(np.sum((t_yzt[:, 2:] - o_yzt[:, 2:]) ** 2, axis=1)) 
        self.delta_t_count += len(delta_t)
        self.delta_t_sum += np.sum(delta_t)
        self.delta_t_sum_sq += np.sum(delta_t**2)  

        # Determine predictions based on distance threshold and temporal consistency
        num_of_predictions = len(delta_yz)
        predicted_triggers = np.zeros(num_of_predictions)
        target_triggers = np.zeros(num_of_predictions)
        for i in range(num_of_predictions):
            if (np.abs(o_yzt[i][0]) <= self.security_distance_mm) and (np.abs(o_yzt[i][1]) <= self.security_distance_mm):
                predicted_triggers[i] = 1

            if (np.abs(t_yzt[i][0]) <= self.security_distance_mm) and (np.abs(t_yzt[i][1]) <= self.security_distance_mm):
                target_triggers[i] = 1

        # Calculate precision, recall, F1 score, and confusion matrix  
        precision = precision_score(target_triggers, predicted_triggers, zero_division=0)
        recall = recall_score(target_triggers, predicted_triggers, zero_division=0)
        f1 = f1_score(target_triggers, predicted_triggers, zero_division=0)
        accuracy = accuracy_score(target_triggers, predicted_triggers)
        conf_matrix = confusion_matrix(target_triggers, predicted_triggers, labels=[True, False])
        tn, fp, fn, tp = conf_matrix.ravel()
        self.precision += precision
        self.recall += recall
        self.f1 += f1
        self.accuracy += accuracy
        self.conf_matrix += conf_matrix
        self.lenght_num += 1


        if dataset_name != "train":
            self.list_of_delta_yz.append(delta_yz)
            self.list_of_delta_t.append(delta_t)
            self.list_of_drone2ball_distance.append(real_distance)
            self.list_of_origin_rec_ids.append(batch["origin"][0])
            self.targets.add_positions(t_yzt)       
            self.predictions.add_positions(o_yzt)       

    def compute_mu_var_std(self, sum_, count, sum_sq):
        if count == 0:
            return 0,0,0
        
        mean = sum_ / count
        variance = (sum_sq / count) - (mean ** 2)
        std = np.sqrt(variance) 
        return mean, variance, std

    def compute_stats(self): 

        self.precision /= self.lenght_num
        self.recall /= self.lenght_num
        self.f1 /= self.lenght_num
        self.accuracy /= self.lenght_num

        if self.is_direct: 
            delta_yz_mean, delta_yz_variance, delta_yz_std = self.compute_mu_var_std(self.delta_yz_sum, self.delta_yz_count, self.delta_yz_sum_sq)
            delta_t_mean, delta_t_variance, delta_t_std = self.compute_mu_var_std(self.delta_t_sum, self.delta_t_count, self.delta_t_sum_sq) 
            
            stats_dict = {
                "metrics": {"conf_matrix": self.conf_matrix, "precision": self.precision, "recall": self.recall, "F1": self.f1, "accuracy": self.accuracy},
                "delta_yz": {"mean": delta_yz_mean, "std": delta_yz_std, "var": delta_yz_variance},
                "delta_t": {"mean": delta_t_mean, "std": delta_t_std, "var": delta_t_variance},
                }
        else:
            
            distances_mean, distances_variance, distances_std = self.compute_mu_var_std(self.distances_sum, self.distances_count, self.distances_sum_sq)
            stats_dict = {
                "metrics": {"conf_matrix": self.conf_matrix, "precision": self.precision, "recall": self.recall, "F1": self.f1, "accuracy": self.accuracy},
                "distances": {"mean": distances_mean, "std": distances_std, "var": distances_variance}, 
                }
            
        return stats_dict

    def __len__(self):
        return self.delta_yz_count

    def reset(self):
        # For distances
        self.delta_yz_count = 0 
        self.delta_yz_sum = 0
        self.delta_yz_sum_sq = 0 

        self.delta_t_count = 0         
        self.delta_t_sum = 0
        self.delta_t_sum_sq = 0  

        # Update running statistics for distances
        self.distances_count = 0
        self.distances_sum = 0
        self.distances_sum_sq = 0 

        # For V Plot
        self.list_of_origin_rec_ids = []
        self.list_of_delta_yz = []
        self.list_of_delta_t = []
        self.list_of_drone2ball_distance = []

        # F1, Recall, etc.
        self.conf_matrix = np.zeros((2,2))
        self.f1 = 0
        self.recall = 0
        self.accuracy = 0
        self.precision = 0
        self.lenght_num = 0

        self.targets.reset()
        self.predictions.reset()
        # self.rgb.reset()
        self.dvs.reset()
        
    def draw_v_plot(self, filename):    

        array_of_origin_rec_ids = torch.concat(self.list_of_origin_rec_ids).cpu().numpy()
        num_recordings = len(np.unique(array_of_origin_rec_ids))  

        array_of_drone2ball_distance = np.hstack(self.list_of_drone2ball_distance)
        array_of_list_of_delta_yz = np.hstack(self.list_of_delta_yz)
        array_of_list_of_delta_t = np.hstack(self.list_of_delta_t) 

        self.make_v_plot(num_recordings, array_of_drone2ball_distance, array_of_origin_rec_ids, array_of_list_of_delta_yz, 'ABS Collision Estimation YZ Error [mm]', filename.replace('.mp4', "_yz_scatter_plot.png"))
        self.make_v_plot(num_recordings, array_of_drone2ball_distance, array_of_origin_rec_ids, array_of_list_of_delta_t, 'ABS TTC Estimation Error [ms]', filename.replace('.mp4', "_t_scatter_plot.png"))

        self.plot_errorbar(filename.replace('.mp4', ".png")) 

    def plot_errorbar(self, filename):

        # Init
        delta_x_target, delta_y_target, delta_t_target =   np.vstack(self.targets.positions)[..., 0],  np.vstack(self.targets.positions)[..., 1],  np.vstack(self.targets.positions)[..., 2]
        delta_x_pred, delta_y_pred, delta_t_pred =   np.vstack(self.predictions.positions)[..., 0],  np.vstack(self.predictions.positions)[..., 1],  np.vstack(self.predictions.positions)[..., 2]

        # Plotting
        plt.close()  
        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        plt.scatter(delta_x_target, delta_x_pred, color='lightblue')
        min_tmp = min(min(delta_x_target), min(delta_x_pred))
        max_tmp = max(max(delta_x_target), max(delta_x_pred))
        plt.plot([min_tmp, max_tmp], [min_tmp, max_tmp], 'r--')  # Line y=x for reference
        plt.xlabel('Target Δx')
        plt.ylabel('Predicted Δx')
        plt.title('Δx: Prediction vs Target')

        plt.subplot(1, 3, 2)
        plt.scatter(delta_y_target, delta_y_pred, color='lightgreen')
        min_tmp = min(min(delta_y_target), min(delta_y_pred))
        max_tmp = max(max(delta_y_target), max(delta_y_pred))
        plt.plot([min_tmp, max_tmp], [min_tmp, max_tmp], 'r--')  # Line y=x for reference
        plt.xlabel('Target Δy')
        plt.ylabel('Predicted Δy')
        plt.title('Δy: Prediction vs Target')

        plt.subplot(1, 3, 3)
        plt.scatter(delta_t_target, delta_t_pred, color='pink')
        min_tmp = min(min(delta_t_target), min(delta_t_pred))
        max_tmp = max(max(delta_t_target), max(delta_t_pred))
        plt.plot([min_tmp, max_tmp], [min_tmp, max_tmp], 'r--')  # Line y=x for reference
        plt.xlabel('Target Δt')
        plt.ylabel('Predicted Δt')
        plt.title('Δt: Prediction vs Target')

        plt.savefig(filename.replace(".png", "_align.png"))
        plt.close()  

        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        plt.hist(delta_x_pred - delta_x_target, bins=20, color='lightblue', alpha=0.7)
        plt.xlabel('Δx Error')
        plt.title('Δx Error Distribution')

        plt.subplot(1, 3, 2)
        plt.hist(delta_y_pred - delta_y_target, bins=20, color='lightgreen', alpha=0.7)
        plt.xlabel('Δy Error')
        plt.title('Δy Error Distribution')

        plt.subplot(1, 3, 3)
        plt.hist(delta_t_pred - delta_t_target, bins=20, color='pink', alpha=0.7)
        plt.xlabel('Δt Error')
        plt.title('Δt Error Distribution')

        plt.savefig(filename.replace(".png", "_dist.png"))
        plt.close()

    def make_v_plot(self, num_recordings, array_of_drone2ball_distance, array_of_origin_rec_ids, array_of_list_of_delta, y_axis_name, filename):
        # Initialize
        plt.close()  
        fig, ax = plt.subplots(figsize=(12, 8)) 

        # Add Scatter Points 
        colormap = plt.cm.get_cmap('tab20', num_recordings) 
        color_map = {rec_id: colormap(i) for i, rec_id in enumerate(np.unique(array_of_origin_rec_ids))} 
        point_colors = np.array([color_map[idx] for idx in array_of_origin_rec_ids])
        ax.scatter(array_of_drone2ball_distance, array_of_list_of_delta, s=10, alpha=0.6, c=point_colors) 
        
        # Add Slop Line
        slope, intercept = np.polyfit(array_of_drone2ball_distance, array_of_list_of_delta, 1)
        ax.plot(array_of_drone2ball_distance, slope * array_of_drone2ball_distance + intercept, color='red', alpha=0.6) 

        # Add Legends, Titles, etc.
        #ax.set_title('')
        ax.set_xlabel('True Distance [mm]')
        ax.set_ylabel(y_axis_name) 
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0, right=2500)
        #ax.set_ylim(bottom=0, top=4500)
        handles = []
        for idx in np.unique(array_of_origin_rec_ids): 
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[idx], markersize=6, label=f'{idx}')) 
        ax.legend(handles=handles, title="Recordings", loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Save Figure
        plt.savefig(filename)
        plt.close()  

    def plot_movement_film(self, file_name):
        if len(self.dvs.data) > 0:   
            dvs_stack = np.vstack(self.dvs.data)  

        # Initialize plot
        plt.close()
        fig = plt.figure(constrained_layout=True, figsize=(32, 12))
        gs = fig.add_gridspec(1, 2)  
        ax_input_2 = fig.add_subplot(gs[0, 0])  
        #ax_target_1 = fig.add_subplot(gs[0, 1], projection='3d')

        ax_yz_2d = fig.add_subplot(gs[0, 1])  
        #ax_target_map = {'targets': ax_target_1, 'predictions': ax_target_1}   
        ax_target_map = {'targets': 0, 'predictions': 0}   

        # Update Function
        def update(frame):  
            # Input axis   
            if len(self.dvs.data) > 0: 
                self.dvs.plot(ax_input_2, {'data': dvs_stack[frame]}, reduce=False)

            # Clear the 2D YZ axis before re-plotting
            ax_yz_2d.clear()
            ax_yz_2d.set_xlabel('Y Axis')
            ax_yz_2d.set_ylabel('Z Axis')
            ax_yz_2d.set_xlim(-1500, 1500)
            ax_yz_2d.set_ylim(-1000, 1000)

            # Trajectories
            ttc = {}
            for key, ax in ax_target_map.items():   
                traj = self.__dict__[key][frame] 
                ttc[key] = np.round(traj[2], 2)
                ax_yz_2d.scatter(traj[0], traj[1], label=key, color=self.__dict__[key].c_point_traj[0], s=100) 
                #self.__dict__[key].plot(ax, traj)     
            
            ax_yz_2d.set_title(f"Collision in {ttc['targets']} vs predicted {ttc['predictions']} <-> Error : {ttc['targets']- ttc['predictions']}")

        # Create Video
        frames = min(300, len(dvs_stack))
        with tqdm(total=frames) as pbar:
            def update_with_progress(frame):
                update(frame)
                pbar.update(1) 

        # Ensure to run the animation saving code on the main thread
        if threading.current_thread() is threading.main_thread():
            ani = FuncAnimation(fig, update_with_progress, frames=frames, blit=False, interval=1)
            ani.save(file_name, writer='ffmpeg', fps=30) 
            plt.close()
        else:
            def save_animation():
                ani = FuncAnimation(fig, update_with_progress, frames=frames, blit=False, interval=1)
                ani.save(file_name, writer='ffmpeg', fps=30) 
                plt.close()
            
            # Use the main thread to execute the saving function
            main_thread = threading.main_thread()
            main_thread.run(save_animation)

    def plot(self, file_name):  
        # Data Initialization
        if self.is_direct and len(self.dvs.data) > 0:
            self.draw_v_plot(file_name)   
            self.plot_movement_film(file_name)

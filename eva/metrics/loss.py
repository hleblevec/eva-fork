import torch
import torch.nn as nn
import pdb 

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class PositionVelocityLoss(nn.Module):
    def __init__(self, multitask_uncertainty_loss=False):
        super().__init__()
        self.mse_loss = nn.MSELoss() 
        self.multitask_uncertainty_loss = multitask_uncertainty_loss
        if multitask_uncertainty_loss:
            self.loss_automatic_weight = AutomaticWeightedLoss(2)

    def forward(self, preds, targets):  
        loss_dict = {"total_loss": 0} 

        if 'delta_drone_2_point_of_collision_yzt' in targets.keys():
            name = 'delta_drone_2_point_of_collision_yzt'
            pos_target, pos_pred = targets[name][:, :2], preds[name][:, :2]  
            loss_dict["loss_delta_drone_2_point_of_collision_yz"] = self.mse_loss(pos_pred, pos_target)

            time_target, time_pred = targets[name][:, 2:], preds[name][:, 2:]  
            loss_dict["loss_delta_drone_2_point_of_collision_t"] = self.mse_loss(time_pred, time_target)
            
            if self.multitask_uncertainty_loss:
                loss_dict["total_loss"] = self.loss_automatic_weight(
                    loss_dict["loss_delta_drone_2_point_of_collision_t"], 
                    loss_dict["loss_delta_drone_2_point_of_collision_yz"]) 
            else:
                loss_dict["total_loss"] += loss_dict["loss_delta_drone_2_point_of_collision_yz"]
                loss_dict["total_loss"] += loss_dict["loss_delta_drone_2_point_of_collision_t"]
                
                    
        if 'delta_ball_2_point_of_collision_yzt' in targets.keys():
            name = 'delta_ball_2_point_of_collision_yzt'
            pos_target = targets[name][:, :2]
            pos_pred = preds[name][:, :2]  
            loss_dict["loss_delta_ball_2_point_of_collision_yz"] = self.mse_loss(pos_pred, pos_target)
            loss_dict["total_loss"] += loss_dict["loss_delta_ball_2_point_of_collision_yz"]

            time_target = targets[name][:, 2:]
            time_pred = preds[name][:, 2:]  
            loss_dict["loss_delta_ball_2_point_of_collision_t"] = self.mse_loss(time_pred, time_target)
            loss_dict["total_loss"] += loss_dict["loss_delta_ball_2_point_of_collision_t"]


        if 'delta_ball_2_drone_3d_xyz' in targets.keys(): 
            name = 'delta_ball_2_drone_3d_xyz'
            ball_3d_target = targets[name]
            ball_3d_pred = preds[name]  
            loss_dict[name] = self.mse_loss(ball_3d_pred, ball_3d_target)
            loss_dict["total_loss"] += loss_dict[name]

        return loss_dict

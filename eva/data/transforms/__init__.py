from .event_transform import EventReduceVoxel, EventAccumulation, EventRandomDrop, EventUniformNoise, SpatialBinning
from .other_transform import Normalize3DPoints, NormalizeRange, NormalizeRangeRobust, NormalizeImage, TensorToNumpy, VICON_DATASET_AGGREGATE, CorrectWhiteBalance
from torchvision.transforms import InterpolationMode

def get_transforms(dvs_res=[320,320], augmentation=False):  
    from torchvision import transforms 
    my_transforms = {
        # targets
        "delta_drone_2_point_of_collision_yzt": transforms.Compose([
            Normalize3DPoints("delta_drone_2_point_of_collision_yzt")]), 
        "delta_ball_2_point_of_collision_yzt": transforms.Compose([
            Normalize3DPoints("delta_ball_2_point_of_collision_yzt")]), 
        "delta_ball_2_drone_3d_xyz": transforms.Compose([
            Normalize3DPoints("delta_ball_2_drone_3d_xyz")]),  
        
        # inputs
        "i_rgb": transforms.Compose([
            transforms.Resize((dvs_res[0], dvs_res[1])),  
            transforms.Lambda(lambda x: x.clamp(0, 255)), 
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Lambda(lambda x: (x - 0.5) * 2)
            ]), 
        "i_dvs": transforms.Compose([
            #transforms.Resize(size=(dvs_res[0], dvs_res[1]), interpolation=InterpolationMode.BICUBIC), 
            SpatialBinning(dvs_res[0], dvs_res[1]),
            EventUniformNoise(noise_level=5) if augmentation else transforms.Lambda(lambda x: x),
            EventRandomDrop(drop_probability=0.2) if augmentation else transforms.Lambda(lambda x: x), 
            NormalizeRange(), 
        ])
    } 
    return my_transforms


def move_tensor_dict_to_device(data_dict, device): 
    for key in data_dict:
        data_dict[key] = data_dict[key].to(device) 
    return data_dict
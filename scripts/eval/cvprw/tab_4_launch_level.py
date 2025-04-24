import pdb, os, torch, yaml, copy
from torch.utils.data import DataLoader
from collections import defaultdict, deque
import numpy as np
import math
from dotenv import load_dotenv

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    accuracy_score,
)
 
from eva.metrics.collisions import CollisionStats
from eva.nn.model import EvaNet
from eva.data.dataset import ABCDataset
from eva.data.utils.utils import load_yaml_from_txt
from eva.data.transforms import TensorToNumpy, get_transforms, move_tensor_dict_to_device

from pytorch_lightning import seed_everything
seed_everything(1234, workers=True)

device = torch.device("cuda")
load_dotenv()

# configs and paths 
base_path = os.path.join(os.getenv("OUTPUT_PATH"), 'results', "dvs-model") 
config = load_yaml_from_txt(os.path.join(base_path, "config.yaml"))

# load model 
model = EvaNet(config["model_config"])
model.load_state_dict(torch.load(os.path.join(base_path, "model.pth")))
model = model.to(device)
model.eval() 

# load the val dataset
dataset_name = "test_dpu"
data_config = config["data_config"]
splits = yaml.safe_load(open(os.path.join(data_config["in_dir"], 'config.yaml'), 'r'))  
indices = splits.get(f'{dataset_name}_indices', [])    
transforms = get_transforms(dvs_res=data_config["dvs_res"])
dataset = ABCDataset(config=data_config, indexes=indices, transforms=transforms)
loader = DataLoader(dataset, num_workers=0, batch_size=1, pin_memory=True, shuffle=False)

# de normalize
task_name = config["data_config"]["outputs_list"][0] 
norm_target_fn =  copy.deepcopy(transforms)[task_name].transforms[0].to_device(device)
tensor_to_numpy_fn = TensorToNumpy()

# Dictionaries to hold predictions per recording
y_true_per_recording = defaultdict(list)
y_pred_per_recording = defaultdict(list)

for step, data_dict in enumerate(loader):
    recording_id, recording_index, original_id = data_dict["origin"] 
    input_dict = move_tensor_dict_to_device(data_dict["inputs"], device)
    target_dict = move_tensor_dict_to_device(data_dict["targets"], device)

    # Model prediction
    outputs = model(**input_dict)
    output_vector = tensor_to_numpy_fn(norm_target_fn.revert(outputs))[0]
    target_vector = tensor_to_numpy_fn(norm_target_fn.revert(target_dict[task_name]))[0]

    # Extract X and Y movement
    predicted_x = output_vector[0]
    predicted_y = output_vector[1]
    predicted_t = output_vector[2]
    actual_x = target_vector[0]
    actual_y = target_vector[1]
    actual_t = target_vector[2]

    # Determine movement classes  
    hor_str = "right" if predicted_x >= 0 else "left"
    ver_str = "up" if predicted_y >= 0 else "down"
    predicted_class = "move_"+ hor_str + "_"+ ver_str

    hor_str = "right" if actual_x >= 0 else "left"
    ver_str = "up" if actual_y >= 0 else "down"
    actual_class = "move_"+ hor_str + "_"+ ver_str 
    
    # Append position-level predictions to the respective recording
    y_true_per_recording[recording_id.item()].append(actual_class)
    y_pred_per_recording[recording_id.item()].append(predicted_class)

# Majority Voting for each recording
final_y_true = []
final_y_pred = []
 
for recording_id in y_true_per_recording:
    majority_true = max(set(y_true_per_recording[recording_id]), key=y_true_per_recording[recording_id].count)
    majority_pred = max(set(y_pred_per_recording[recording_id]), key=y_pred_per_recording[recording_id].count)
    
    final_y_true.append(majority_true)
    final_y_pred.append(majority_pred)
 

# Compute accuracy and F1 scores on the recording level
accuracy = accuracy_score(final_y_true, final_y_pred)
f1 = f1_score(final_y_true, final_y_pred, average="weighted")

print(f"Overall Accuracy: {accuracy:.2f}")
print(f"Overall F1 Score: {f1:.2f}")

# Compute per-class accuracy and F1 scores
classes = ["move_left_up", "move_right_up", "move_left_down", "move_right_down"]
for class_label in classes:
    class_accuracy = accuracy_score(
        [1 if label == class_label else 0 for label in final_y_true],
        [1 if label == class_label else 0 for label in final_y_pred]
    )
    class_f1 = f1_score(
        [1 if label == class_label else 0 for label in final_y_true],
        [1 if label == class_label else 0 for label in final_y_pred],
        zero_division=0
    )
    class_precision = precision_score(
        [1 if label == class_label else 0 for label in final_y_true],
        [1 if label == class_label else 0 for label in final_y_pred],
        zero_division=0
    )
    class_recall = recall_score(
        [1 if label == class_label else 0 for label in final_y_true],
        [1 if label == class_label else 0 for label in final_y_pred],
        zero_division=0
    )
    print(f"{class_label} Accuracy: {class_accuracy:.2f}")
    print(f"{class_label} F1 Score: {class_f1:.2f}")
    print(f"{class_label} Recall: {class_recall:.2f}")
    print(f"{class_label} Precision: {class_precision:.2f}")
    print("-------------------------------------------------")


# Compute confusion matrix
conf_matrix = confusion_matrix(final_y_true, final_y_pred, labels=classes)

# Plot and save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(base_path, "confusion_matrix.png"))


# Calculate recall (True Positive Rate) for each class
class_recalls = []
for i, class_label in enumerate(classes):
    tp = conf_matrix[i, i]  # True positives for class
    fn = conf_matrix[i, :].sum() - tp  # False negatives for class
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Avoid division by zero
    class_recalls.append(recall)

# Calculate Balanced Accuracy (average of recall for each class)
balanced_accuracy = np.mean(class_recalls)

print(f"Balanced Accuracy: {balanced_accuracy:.2f}")
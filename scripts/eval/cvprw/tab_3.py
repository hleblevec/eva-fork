import os, torch, yaml, copy
from torch.utils.data import DataLoader
import numpy as np
from dotenv import load_dotenv

from sklearn.metrics import f1_score
from pytorch_lightning import seed_everything
seed_everything(1234, workers=True)

from eva.nn.model import EvaNet
from eva.data.dataset import ABCDataset
from eva.data.utils.utils import load_yaml_from_txt
from eva.data.transforms import TensorToNumpy, get_transforms, move_tensor_dict_to_device

# Environment + device setup
load_dotenv()
device = torch.device("cuda")
tensor_to_numpy_fn = TensorToNumpy()

# Globals for paths
model_paths = {
    "RGB": os.path.join(os.getenv("OUTPUT_PATH"), "results", "rgb-model"),
    "EVS": os.path.join(os.getenv("OUTPUT_PATH"), "results", "dvs-model"),
}
datasets = [("1", "test"), ("2", "test_dpu")]

# Cache configs and transforms
cached = {}
for model_type, model_path in model_paths.items():
    config = load_yaml_from_txt(os.path.join(model_path, "config.yaml"))
    data_config = config["data_config"]
    transform = get_transforms(dvs_res=data_config["dvs_res"])
    task_name = data_config["outputs_list"][0]
    norm_target_fn = copy.deepcopy(transform)[task_name].transforms[0].to_device(device)
    cached[model_type] = {
        "config": config,
        "transform": transform,
        "task_name": task_name,
        "norm_target_fn": norm_target_fn,
    }

def load_dataset(data_config, dataset_name, transform):
    split_path = os.path.join(data_config["in_dir"], "config.yaml")
    splits = yaml.safe_load(open(split_path, "r"))
    indices = splits.get(f"{dataset_name}_indices", [])
    dataset = ABCDataset(config=data_config, indexes=indices, transforms=transform)
    loader = DataLoader(dataset, num_workers=0, batch_size=1, pin_memory=True, shuffle=False)
    return loader

def evaluate(model, loader, task_name, norm_target_fn):
    correct_x = correct_y = correct_t = total = 0
    y_true_x, y_pred_x = [], []
    y_true_y, y_pred_y = [], []
    y_true_t, y_pred_t = [], []

    for data_dict in loader:
        input_dict = move_tensor_dict_to_device(data_dict["inputs"], device)
        target_dict = move_tensor_dict_to_device(data_dict["targets"], device)

        with torch.no_grad():
            output = model(**input_dict)

        pred = tensor_to_numpy_fn(norm_target_fn.revert(output))[0]
        target = tensor_to_numpy_fn(norm_target_fn.revert(target_dict[task_name]))[0]

        pred_dirs = [np.sign(pred[0]), np.sign(pred[1]), pred[2] < 100]
        true_dirs = [np.sign(target[0]), np.sign(target[1]), target[2] < 100]

        correct_x += pred_dirs[0] == true_dirs[0]
        correct_y += pred_dirs[1] == true_dirs[1]
        correct_t += pred_dirs[2] == true_dirs[2]
        total += 1

        y_true_x.append(true_dirs[0]); y_pred_x.append(pred_dirs[0])
        y_true_y.append(true_dirs[1]); y_pred_y.append(pred_dirs[1])
        y_true_t.append(true_dirs[2]); y_pred_t.append(pred_dirs[2])

    acc_x = round(100 * correct_x / total)
    acc_y = round(100 * correct_y / total)
    acc_t = round(100 * correct_t / total)

    f1_x = round(f1_score(y_true_x, y_pred_x, average="weighted"), 2)
    f1_y = round(f1_score(y_true_y, y_pred_y, average="weighted"), 2)
    f1_t = round(f1_score(y_true_t, y_pred_t, average="weighted"), 2)

    return acc_x, f1_x, acc_y, f1_y, acc_t, f1_t

# Run evaluations
results = []

for dataset_label, dataset_name in datasets:
    for model_type, model_path in model_paths.items():
        conf = cached[model_type]
        model = EvaNet(conf["config"]["model_config"])
        model.load_state_dict(torch.load(os.path.join(model_path, "model.pth")))
        model = model.to(device).eval()

        loader = load_dataset(conf["config"]["data_config"], dataset_name, conf["transform"])
        metrics = evaluate(model, loader, conf["task_name"], conf["norm_target_fn"])
        if model_type == "RGB":
            dataset_label = "Test"
        results.append((dataset_label, model_type) + metrics)

# Print LaTeX table
print(r"""
\begin{table}
\centering
\resizebox{\linewidth}{!}{%
\begin{tabular}{c | c | cc | cc | cc}
\toprule
\multirow{2}{*}{\textbf{Data}} & \multirow{2}{*}{\textbf{Model}} & \multicolumn{2}{c|}{\textbf{Left/Right}} & \multicolumn{2}{c|}{\textbf{Up/Down}} & \multicolumn{2}{c}{\textbf{Move/Stay}}  \\ 
 &  & \textbf{Acc} & \textbf{F1} & \textbf{Acc} & \textbf{F1} & \textbf{Acc} & \textbf{F1}  \\ 
\midrule
""")

for i, (dataset, model, acc_x, f1_x, acc_y, f1_y, acc_t, f1_t) in enumerate(results):
    print(f"{dataset} & {model} & {acc_x} & {f1_x} & {acc_y} & {f1_y} & {acc_t} & {f1_t} \\\\ ", end="")
    print("  \\cmidrule{2-8}" if model == "RGB" else "  \\midrule" if i < len(results) - 1 else "  \\bottomrule")

print(r"""\end{tabular}%
}
\caption{\textbf{Comparison of collision prediction models.} Individual window-level class evaluations for two test sets.}
\label{tab:rgb_vs_evs_results}
\end{table}
""")

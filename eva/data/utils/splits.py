import pdb
import os
import yaml
import numpy as np
from dtaidistance import dtw_ndim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed

def euclidean_distance(traj1, traj2):
    assert traj1.shape == traj2.shape, "Trajectories must have the same shape"  
    distances = np.linalg.norm(traj1 - traj2, axis=1) 
    return np.mean(distances)

def dtw_distance(traj1, traj2):
    assert traj1.shape == traj2.shape, "Trajectories must have the same shape"
    distance = dtw_ndim.distance(traj1, traj2)
    return distance

class TrajectorySimilarity:
    def __init__(self, dataset, out_dir, num_workers, method):
        self.dataset = dataset
        self.trajectories = None
        self.similarity_matrix = None
        self.method = method
        self.out_file = os.path.join(out_dir, f"similarity_matrix_{self.method}.npy")
        self.num_workers = num_workers

    def _compute_similarity_matrix(self):
        self._collect_trajectories()
        num_trajectories = len(self.trajectories)
        similarity_matrix = np.zeros((num_trajectories, num_trajectories))

        def compute_distance(i, j):
            if i == j:
                return 0
            if self.method == "euclidean":
                return euclidean_distance(self.trajectories[i], self.trajectories[j])
            elif self.method == "dtw":
                return dtw_distance(self.trajectories[i], self.trajectories[j])

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(compute_distance, i, j): (i, j)
                       for i in range(num_trajectories) for j in range(i+1, num_trajectories)}
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    i, j = futures[future]
                    dist = future.result()
                    similarity_matrix[i, j] = dist
                    similarity_matrix[j, i] = dist
                except Exception as exc:
                    print('Generated exception: %s' % (exc))  



        self.similarity_matrix = similarity_matrix
        np.save(self.out_file, similarity_matrix)

    def compute_similarity_matrix(self):
        if os.path.exists(self.out_file):
            self.similarity_matrix = np.load(self.out_file)
        if self.similarity_matrix is None:
            self._compute_similarity_matrix()

    def _collect_trajectories(self):
        self.trajectories = []
        traj_length = 53

        for i in tqdm(range(self.dataset.n_of_indexes)):  
            traj = self.dataset.dataset[i].ball.positions[:traj_length, :]
            if traj.shape[0] != traj_length: 
                missing_points = traj_length - traj.shape[0]
                end_points = traj[-1:, :]
                repeated_end_points = np.tile(end_points, (missing_points, 1))
                traj = np.concatenate((traj, repeated_end_points), axis=0)

            self.trajectories.append(traj)

    def visualize_similarity_matrix(self):
        self.compute_similarity_matrix()
        plt.imshow(self.similarity_matrix, cmap='RdBu', origin='lower')
        plt.colorbar(label=f'Magnitude')
        plt.title(f'Similarity Matrix : {self.method}')
        plt.xlabel('Trajectory Index')
        plt.ylabel('Trajectory Index')
        plt.savefig(f'./output/similarity_plot_{self.method}.pdf')

if __name__ == '__main__':
    from ..dataset import GLCDataset
    from ..transforms import get_transforms

    method = "dtw"

    data_config = {
        "in_dir": "/datasets/pbonazzi/sony-rap/glc_dataset/vicon_aggregate/",
        "num_workers": 10,
        "batch_size": 8,
        "event_dt_ms": 40,
        "rgb_windows": 1,
        "event_windows": 1,
        "event_polarities": 1,
        "event_polarities_mode": "substract",
        "event_accumulation": "addition",
        "event_decay_constant": 0.1,
        "imu_windows": 1,
        "frequency": "dvs",
        "max_num_of_e": 500,
        "overwrite": False,
        "outputs_list": ["delta_ball_2_drone__3d_xyz"],
        "inputs_list": ["dvs"]
    } 

    # range_indexes = list(range(1, 270)) 
    # to_remove = [4, 8, 13, 14, 15, 16, 17, 18, 22, 31, 38, 54, 72, 73, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 104, 113, 125, 134, 141, 149, 168, 169, 170, 187, 191, 206, 213, 214, 229, 230, 249, 251, 266 ]
    # true_indexes = [idx for idx in range_indexes if idx not in to_remove]

    import yaml
    from torchvision.transforms import InterpolationMode

    splits = yaml.safe_load(open(os.path.join(data_config["in_dir"], 'config.yaml'), 'r')) 
    train_indices = splits.get('train_indices', [])
    val_indices = splits.get('val_indices', [])
    test_indices = splits.get('test_indices', [])
    outlier_indices = splits.get('outlier_indices', [])
    true_indexes = train_indices + val_indices + test_indices + outlier_indices

    # range_indexes = list(range(1, 270)) 
    # to_remove = [4, 8, 13, 14, 15, 16, 17, 18, 22, 31, 38, 54, 72, 73, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 104, 113, 125, 134, 141, 149, 168, 169, 170, 187, 191, 206, 213, 214, 229, 230, 249, 251, 266 ]
    # true_indexes = [idx for idx in range_indexes if idx not in to_remove]

    dataset = GLCDataset(config=data_config, indexes=true_indexes, transforms=get_transforms())
    similarity_calculator = TrajectorySimilarity(dataset, data_config["in_dir"], data_config["num_workers"], method)
    similarity_calculator.visualize_similarity_matrix()

    # k = 8
    # kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    # clusters = kmeans.fit_predict(similarity_calculator.similarity_matrix)

    # train_indices, val_indices, test_indices = [], [], []

    # for cluster_id in range(k):
    #     cluster_indices = np.where(clusters == cluster_id)[0]
    #     if len(cluster_indices) < 4:
    #         train_indices.extend(cluster_indices.tolist())
    #     else:
    #         cluster_train_val, cluster_test = train_test_split(cluster_indices, test_size=0.15, random_state=42)
    #         cluster_train, cluster_val = train_test_split(cluster_train_val, test_size=0.15, random_state=42)
    #         train_indices.extend(cluster_train.tolist())
    #         val_indices.extend(cluster_val.tolist())
    #         test_indices.extend(cluster_test.tolist())

    # train_indices = [true_indexes[i] for i in train_indices]
    # val_indices = [true_indexes[i] for i in val_indices]
    # test_indices = [true_indexes[i] for i in test_indices]

    # config = {
    #     'train_indices': train_indices,
    #     'val_indices': val_indices,
    #     'test_indices': test_indices
    # }

    # with open(os.path.join(data_config["in_dir"], 'config.yaml'), 'w') as f:
    #     yaml.dump(config, f)

    # print("Saved Config File")

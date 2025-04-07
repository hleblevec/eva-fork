import pdb
import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    accuracy_score,
)

class CollisionStats:
    def __init__(self): 
        
        # Initialize cumulative metrics
        self.recording_metrics = defaultdict(list)  # Stores metrics for each recording


    def update_stats(self, recording_id, target_triggers, predicted_triggers): 
        
        if len(targets) != len(outputs):
            raise ValueError("Targets and outputs must have the same length")
        
        # Calculate metrics (handling zero division gracefully)
        precision = precision_score(target_triggers, predicted_triggers, zero_division=0)
        recall = recall_score(target_triggers, predicted_triggers, zero_division=0)
        f1 = f1_score(target_triggers, predicted_triggers, zero_division=0)
        accuracy = accuracy_score(target_triggers, predicted_triggers)
        conf_matrix = confusion_matrix(target_triggers, predicted_triggers, labels=[True, False])
        
        # Store metrics by recording ID
        self.recording_metrics[recording_id].append({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix
        })


    def compute_recording_level_metrics(self):
        recording_level_results = {}
        for rec_id, metrics_list in self.recording_metrics.items():
            # Aggregate metrics for each recording
            total_conf_matrix = np.sum([m["confusion_matrix"] for m in metrics_list], axis=0)
            avg_precision = np.mean([m["precision"] for m in metrics_list])
            avg_recall = np.mean([m["recall"] for m in metrics_list])
            avg_f1 = np.mean([m["f1"] for m in metrics_list])
            avg_accuracy = np.mean([m["accuracy"] for m in metrics_list])

            recording_level_results[rec_id] = {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1,
                "accuracy": avg_accuracy,
                "confusion_matrix": total_conf_matrix
            }
        return recording_level_results


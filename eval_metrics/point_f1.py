from sklearn.metrics import precision_recall_curve
import numpy as np
import json

def calculate_point_f1(scores: np.ndarray, labels: np.ndarray):
    precision, recall, thresholds = precision_recall_curve(labels, scores)

    # Compute F1 scores for all thresholds
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # Avoid division by zero

    # Find the threshold that maximizes the F1 score
    best_idx = np.nanargmax(f1_scores)  # Ignore NaNs
    best_f1 = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_threshold = thresholds[best_idx]

    # Return results in JSON format
    result = {
        "f1": best_f1,
        "precision": best_precision,
        "recall": best_recall,
        "threshold": float(best_threshold)
    }
    
    return result

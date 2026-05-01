# Removed all but one rule to comply with the requirement

import numpy as np

def inference(sample: np.ndarray) -> np.ndarray:
    # Normal Rule 1: Data should not have prolonged increasing or decreasing trends (>50 units) for 4 consecutive points.
    
    labels = np.zeros(sample.shape[0], dtype=int)

    if sample.shape[0] < 4:
        return labels

    # Abnormal Rule 1: Detect abnormal trends: consecutive increases or decreases (>50 units) for 4 points
    for i in range(3, sample.shape[0]):
        window = sample[i-3:i+1, 0]
        diffs = np.diff(window)

        if np.all(diffs > 50) or np.all(diffs < -50):
            labels[i] = 1
            break

    return labels

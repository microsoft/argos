# Removed Abnormal Rule 3 as it was redundant with Abnormal Rule 1 and 2

import numpy as np

def inference(sample: np.ndarray) -> np.ndarray:
    # Normal Rule 1: Data should not have prolonged increasing or decreasing trends (>50 units) for 4 consecutive points.
    # Normal Rule 2: Differences between consecutive points should generally be less than 200 units.

    labels = np.zeros(sample.shape[0], dtype=int)

    if sample.shape[0] < 4:
        return labels

    mean = np.mean(sample[:, 0])
    std_dev = np.std(sample[:, 0])

    # Abnormal Rule 1: Detect abnormal trends: consecutive increases or decreases (>50 units) for 4 points
    for i in range(3, sample.shape[0]):
        window = sample[i-3:i+1, 0]
        diffs = np.diff(window)

        if np.all(diffs > 50) or np.all(diffs < -50):
            labels[i] = 1
            break

    
    # Abnormal Rule 2: Detect abnormal large jumps between consecutive points (>200 units)
    consecutive_diffs = np.abs(np.diff(sample[:, 0]))
    abnormal_jump_indices = np.where(consecutive_diffs > 200)[0] + 1
    labels[abnormal_jump_indices] = 1

    return labels
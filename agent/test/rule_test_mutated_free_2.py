# Mutation: Added a new Normal Rule 4 and modified Abnormal Rule 3 to account for the new normal behavior
import numpy as np

def inference(sample: np.ndarray) -> np.ndarray:
    # Normal Rule 1: Data should not have prolonged increasing or decreasing trends (>50 units) for 4 consecutive points.
    # Normal Rule 2: Differences between consecutive points should generally be less than 200 units.
    # Normal Rule 3: The last 4 data points should stay within ±20% of the overall mean.
    # Normal Rule 4: Data should have at least one point within ±10% of the mean in every 50 points.

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

    # Abnormal Rule 3: Check if the last 4 values deviate more than 20% from the mean
    last_four = sample[-4:, 0]
    lower_bound = mean * 0.8
    upper_bound = mean * 1.2

    if np.all(last_four < lower_bound) or np.all(last_four > upper_bound):
        labels[-1] = 1

    # Abnormal Rule 4: Check for absence of points within ±10% of the mean in any 50-point window
    window_size = 50
    for i in range(0, sample.shape[0] - window_size + 1):
        window = sample[i:i+window_size, 0]
        window_mean = np.mean(window)
        lower_window = window_mean * 0.9
        upper_window = window_mean * 1.1
        if np.all(window < lower_window) or np.all(window > upper_window):
            labels[i:i+window_size] = 1
            break

    return labels

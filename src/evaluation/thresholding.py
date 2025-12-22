import numpy as np
from sklearn.metrics import balanced_accuracy_score

def optimize_threshold(y_true, y_prob):
    best_thr, best_val = 0.5, -1
    for thr in np.linspace(0.01, 0.99, 200):
        val = balanced_accuracy_score(y_true, y_prob >= thr)
        if val > best_val:
            best_thr, best_val = thr, val
    return best_thr

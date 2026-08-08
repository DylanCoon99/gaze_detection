# Accuracy metrics — MAE, p50/p95/p99 angular error

'''
accuracy.py — Takes two arrays (predictions and ground truth, both yaw/pitch in degrees) and computes: mean absolute error, and percentile errors (p50, p95,
p99). These can be computed per-angle (yaw MAE, pitch MAE separately) and combined. Straightforward numpy math — absolute differences, then np.mean and
np.percentile.
'''

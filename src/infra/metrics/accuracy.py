# Accuracy metrics — MAE, p50/p95/p99 angular error
import numpy as np
from dataclasses import dataclass, field


'''
accuracy.py — Takes two arrays (predictions and ground truth, both yaw/pitch in degrees) and computes: mean absolute error, and percentile errors (p50, p95,
p99). These can be computed per-angle (yaw MAE, pitch MAE separately) and combined. Straightforward numpy math — absolute differences, then np.mean and
np.percentile.
'''

@dataclass
class Accuracy:
	mae_yaw: np.ndarray
	mae_pitch: np.ndarray
	mae_combined: np.ndarray
	percentile_errors_yaw: np.ndarray
	percentile_errors_pitch: np.ndarray



def get_accuracy(y_pred, y_true) -> Accuracy:
	dif = np.abs(y_true - y_pred)
	mae_yaw, mae_pitch = np.mean(dif, axis=0)
	mae_combined = np.mean(dif)
	percentiles = np.percentile(dif, [50, 95, 99], axis=0)                                                                                                         
	percentile_errors_yaw = percentiles[:, 0]                                                                                                                      
	percentile_errors_pitch = percentiles[:, 1] 
	return Accuracy(
		mae_yaw=mae_yaw,
		mae_pitch=mae_pitch,
		mae_combined=mae_combined,
		percentile_errors_yaw=percentile_errors_yaw,
		percentile_errors_pitch=percentile_errors_pitch
	)
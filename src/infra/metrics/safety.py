# Safety metrics — eyes-off-road FPR/FNR given a gaze-cone threshold
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix

'''
Takes the same predictions/ground truth and a threshold angle (e.g. 15°). Classifies each sample as "eyes on road" or "eyes off road" based on
whether the angle exceeds the threshold. Then compares the model's classification against ground truth to compute false positive rate (model says looking away
but driver isn't) and false negative rate (model says fine but driver is looking away). This is the metric that matters for a real product — false positives
make drivers disable the system.
'''


@dataclass
class Safety:
	fpr: float
	fnr: float

def get_safety(y_pred, y_true, threshold_angle=15) -> Safety:

	# compare the y_pred with the threshold
	pred_off_road = (np.abs(y_pred[:, 0]) > threshold_angle) | (np.abs(y_pred[:, 1]) > threshold_angle)
	# compaare y_true with the threshold
	true_off_road = (np.abs(y_true[:, 0]) > threshold_angle) | (np.abs(y_true[:, 1]) > threshold_angle)
	# false positive: model says looking away but driver isn't
	tn, fp, fn, tp = confusion_matrix(true_off_road, pred_off_road, labels=[False, True]).ravel()



	# Calculate FPR
	fpr = (fp / (fp + tn)) if (fp + tn != 0) else 0.0

	# false negative: model says not looking away but driver is
	fnr = (fn / (tp + fn)) if (tp + fn != 0) else 0.0

	return Safety(fpr=fpr, fnr=fnr)
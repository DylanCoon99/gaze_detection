# Slicing — break metrics out by yaw bin (0-15, 15-30, 30-45, 45+)
from dataclasses import dataclass

import numpy as np

from accuracy import get_accuracy, Accuracy

'''
Takes predictions, ground truth, and the ground truth yaw values. Bins the samples by absolute yaw angle (e.g. 0-15°, 15-30°, 30-45°, 45+°). For
each bin, runs the accuracy metrics on just that subset. The output is a dict mapping each bin to its metrics. This is what reveals "the model is great overall
but falls apart at extreme yaw."
'''

@dataclass
class Slicing:
	zero_to_fifteen: Accuracy
	fifteen_to_thirty: Accuracy
	thirty_to_forty_five: Accuracy
	forty_five_plus: Accuracy



def get_slicing(y_pred, y_true):

	# get the ground truth yaw values
	# y_true: [[yaw_true, pitch_true], [yaw_true, pitch_true], ..., [yaw_true, pitch_true]]

	# bin samples by absolute yaw angle
	yaw_angle = np.abs(y_true[:, 0])
	zero_to_fifteen_bin  = y_true[(yaw_angle >= 0) & (yaw_angle < 15)]
	zero_to_fifteen_pred = y_pred[(yaw_angle >= 0) & (yaw_angle < 15)]

	fifteen_to_thirty_bin  = y_true[(yaw_angle >= 15) & (yaw_angle < 30)]
	fifteen_to_thirty_pred = y_pred[(yaw_angle >= 15) & (yaw_angle < 30)]

	thirty_to_forty_five_bin  = y_true[(yaw_angle >= 30) & (yaw_angle < 45)]
	thirty_to_forty_five_pred = y_pred[(yaw_angle >= 30) & (yaw_angle < 45)]

	forty_five_plus_bin  = y_true[yaw_angle >= 45]
	forty_five_plus_pred = y_pred[yaw_angle >= 45]

	# for each bin run the accuracy metrics on the subset
	zero_to_fifteen      = get_accuracy(zero_to_fifteen_pred, zero_to_fifteen_bin)
	fifteen_to_thirty    = get_accuracy(fifteen_to_thirty_pred, fifteen_to_thirty_bin)
	thirty_to_forty_five = get_accuracy(thirty_to_forty_five_pred, thirty_to_forty_five_bin)
	forty_five_plus      = get_accuracy(forty_five_plus_pred, forty_five_plus_bin)

	return Slicing(
		zero_to_fifteen=zero_to_fifteen,
		fifteen_to_thirty=fifteen_to_thirty,
		thirty_to_forty_five=thirty_to_forty_five,
		forty_five_plus=forty_five_plus
	)



# Report — takes predictions + ground truth, runs all metrics, returns structured result
from dataclasses import dataclass

from accuracy import get_accuracy, Accuracy
from latency import get_latency, Latency
from safety import get_safety, Safety
from slicing import get_slicing, Slicing

'''
The entry point. Takes predictions and ground truth, calls accuracy, safety, and slicing, and assembles everything into one structured result (a
dataclass or dict). This is what the runner calls — one function in, one report out. This result is what gets logged to MLflow later.
'''


@dataclass
class Report:
	accuracy: Accuracy
	safety: Safety
	slicing: Slicing
	latency: Latency


# needs to take predictions and labels as input; called by the runner
def report(y_pred, y_true, threshold=15, batch_times_ms=None):

	# call accuracy
	accuracy = get_accuracy(y_pred, y_true)
	# call safety
	safety = get_safety(y_pred, y_true, threshold_angle=threshold)
	# call slicing
	slicing = get_slicing(y_pred, y_true)
	# call latency
	latency = get_latency(batch_times_ms or [])

	# return the Report object
	return Report(accuracy=accuracy, safety=safety, slicing=slicing, latency=latency)
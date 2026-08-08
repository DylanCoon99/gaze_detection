# Report — takes predictions + ground truth, runs all metrics, returns structured result
from dataclasses import dataclass, field
from accuracy import get_accuracy, Accuracy
from safety import get_safety
from slicing import get_slicing

'''
The entry point. Takes predictions and ground truth, calls accuracy, safety, and slicing, and assembles everything into one structured result (a
dataclass or dict). This is what the runner calls — one function in, one report out. This result is what gets logged to MLflow later.
'''


@dataclass
class Report:
	accuracy: Accuracy




# needs to take predictions and labels as input; called by the runner
def report(y_pred, y_true, threshold=15): # threshold should probably some from the model config or passed as an argument when the runner is ran

	# call accuracy
	accuracy = get_accuracy(y_pred, y_true)
	# call safety
	safety = get_safety(y_pred, y_true, threshold=threshold) 
	# call slicing
	
	# return the Report object

	return Report(accuracy=accuracy)
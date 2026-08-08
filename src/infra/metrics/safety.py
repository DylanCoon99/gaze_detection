# Safety metrics — eyes-off-road FPR/FNR given a gaze-cone threshold


'''
Takes the same predictions/ground truth and a threshold angle (e.g. 15°). Classifies each sample as "eyes on road" or "eyes off road" based on
whether the angle exceeds the threshold. Then compares the model's classification against ground truth to compute false positive rate (model says looking away
but driver isn't) and false negative rate (model says fine but driver is looking away). This is the metric that matters for a real product — false positives
make drivers disable the system.
'''

def get_safety(y_pred, y_true, threshold_angle=15):


	return
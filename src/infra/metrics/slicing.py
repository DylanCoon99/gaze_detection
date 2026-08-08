# Slicing — break metrics out by yaw bin (0-15, 15-30, 30-45, 45+)


'''
Takes predictions, ground truth, and the ground truth yaw values. Bins the samples by absolute yaw angle (e.g. 0-15°, 15-30°, 30-45°, 45+°). For
each bin, runs the accuracy metrics on just that subset. The output is a dict mapping each bin to its metrics. This is what reveals "the model is great overall
but falls apart at extreme yaw."
'''
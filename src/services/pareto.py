# Pareto frontier calculation.
# Given a list of models with two metrics, returns only the Pareto-optimal set
# (models where no other model is better on both axes simultaneously).


def pareto_frontier(models, metric_a, metric_b):
	"""Filter models to only those on the Pareto frontier.

	Both metrics are assumed to be "lower is better".
	metric_a and metric_b are dot-notation keys (e.g. "accuracy.mae_combined").
	"""
	def get_value(model, key):
		parts = key.split(".")
		val = model
		for p in parts:
			val = val[p]
		return val

	# Filter out models with missing values for either metric
	models = [m for m in models if get_value(m, metric_a) is not None and get_value(m, metric_b) is not None]

	# Sort by metric_a ascending
	sorted_models = sorted(models, key=lambda m: get_value(m, metric_a))

	frontier = []
	best_b = float("inf")

	for model in sorted_models:
		val_b = get_value(model, metric_b)
		if val_b < best_b:
			frontier.append(model)
			best_b = val_b

	return frontier

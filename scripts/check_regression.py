"""Compare eval results against a baseline and fail if any metric regresses."""
import argparse
import json
import sys

# Maximum allowed regression per metric (positive = worse)
TOLERANCES = {
	"accuracy.mae_yaw": 0.5,
	"accuracy.mae_pitch": 0.5,
	"accuracy.mae_combined": 0.5,
	"safety.fpr": 0.01,
	"safety.fnr": 0.02,
	"slicing.mae_0_15": 0.5,
	"slicing.mae_15_30": 0.5,
	"slicing.mae_30_45": 0.5,
	"slicing.mae_45_plus": 0.5,
}


def get_nested(d, key):
	"""Get a value from a nested dict using dot notation."""
	parts = key.split(".")
	for part in parts:
		d = d[part]
	return d


def main():
	parser = argparse.ArgumentParser(description="Check for metric regressions against a baseline")
	parser.add_argument("--baseline", required=True, help="Path to baseline JSON")
	parser.add_argument("--results", required=True, help="Path to new results JSON")
	args = parser.parse_args()

	with open(args.baseline) as f:
		baseline = json.load(f)
	with open(args.results) as f:
		results = json.load(f)

	failed = False
	for metric, tolerance in TOLERANCES.items():
		base_val = get_nested(baseline, metric)
		new_val = get_nested(results, metric)
		delta = new_val - base_val

		if delta > tolerance:
			print(f"FAIL  {metric}: {base_val:.4f} -> {new_val:.4f} (delta={delta:+.4f}, tolerance={tolerance})")
			failed = True
		else:
			print(f"PASS  {metric}: {base_val:.4f} -> {new_val:.4f} (delta={delta:+.4f})")

	if failed:
		print("\nRegression detected. Blocking merge.")
		sys.exit(1)
	else:
		print("\nAll metrics within tolerance.")


if __name__ == "__main__":
	main()

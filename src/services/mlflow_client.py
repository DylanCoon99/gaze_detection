# Wrapper around MLflow's tracking client.
# Queries the gaze-eval experiment for runs, params, and metrics.
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

EXPERIMENT_NAME = "gaze-eval"
TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]

mlflow.set_tracking_uri(TRACKING_URI)


def _run_to_dict(run):
	"""Convert an MLflow run to a flat dictionary of params + metrics."""
	return {
		"run_id": run.info.run_id,
		"model_name": run.data.params.get("model_name"),
		"weights": run.data.params.get("weights"),
		"runtime": run.data.params.get("runtime"),
		"quantization": run.data.params.get("quantization"),
		"input_resolution": run.data.params.get("input_resolution"),
		"batch_size": run.data.params.get("batch_size"),
		"threshold_angle": run.data.params.get("threshold_angle"),
		"dataset_version": run.data.params.get("dataset_version"),
		"num_samples": run.data.params.get("num_samples"),
		"accuracy": {
			"mae_yaw": run.data.metrics.get("mae_yaw"),
			"mae_pitch": run.data.metrics.get("mae_pitch"),
			"mae_combined": run.data.metrics.get("mae_combined"),
			"p50_yaw": run.data.metrics.get("p50_yaw"),
			"p95_yaw": run.data.metrics.get("p95_yaw"),
			"p99_yaw": run.data.metrics.get("p99_yaw"),
			"p50_pitch": run.data.metrics.get("p50_pitch"),
			"p95_pitch": run.data.metrics.get("p95_pitch"),
			"p99_pitch": run.data.metrics.get("p99_pitch"),
		},
		"safety": {
			"fpr": run.data.metrics.get("fpr"),
			"fnr": run.data.metrics.get("fnr"),
		},
		"slicing": {
			"mae_0_15": run.data.metrics.get("mae_slice_0_15"),
			"mae_15_30": run.data.metrics.get("mae_slice_15_30"),
			"mae_30_45": run.data.metrics.get("mae_slice_30_45"),
			"mae_45_plus": run.data.metrics.get("mae_slice_45_plus"),
		},
	}


def get_latest_per_model():
	"""Return the most recent run for each distinct model."""
	runs = mlflow.search_runs(
		experiment_names=[EXPERIMENT_NAME],
		order_by=["start_time DESC"],
		output_format="list",
	)
	latest = {}
	for r in runs:
		name = r.data.params.get("model_name")
		if name and name not in latest:
			latest[name] = _run_to_dict(r)
	return list(latest.values())


def get_all_runs():
	"""Return all runs in the gaze-eval experiment."""
	runs = mlflow.search_runs(
		experiment_names=[EXPERIMENT_NAME],
		output_format="list",
	)
	return [_run_to_dict(r) for r in runs]


def get_runs_by_model(model_name: str):
	"""Return all runs for a specific model."""
	runs = mlflow.search_runs(
		experiment_names=[EXPERIMENT_NAME],
		filter_string=f"params.model_name = '{model_name}'",
		output_format="list",
	)
	return [_run_to_dict(r) for r in runs]

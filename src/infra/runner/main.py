# Model Evaluation Runner
import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.io as tv_io
import mlflow
from torchvision import transforms

from config import Config
from errors import MalformedConfig
from factory import get_model

# Add metrics to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "metrics"))
from report import report

logger = logging.getLogger("runner")


def load_labels(labels_csv: str):
	"""Load ground truth labels from the preprocessing pipeline's CSV."""
	filenames = []
	labels = []
	with open(labels_csv, encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			filenames.append(row["filename"])
			labels.append([float(row["yaw"]), float(row["pitch"])])
	return filenames, np.array(labels)


def load_images(crop_dir: str, filenames: list, input_resolution: list):
	"""Load preprocessed crop images as tensors."""
	transform = transforms.Compose([
		transforms.Resize(input_resolution),
		transforms.ConvertImageDtype(torch.float32),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	images = []
	for fname in filenames:
		img_path = Path(crop_dir) / fname
		img = tv_io.read_image(str(img_path))
		img = transform(img)
		images.append(img)

	return torch.stack(images)


def main():
	parser = argparse.ArgumentParser(description="Run model evaluation")
	parser.add_argument("--config", type=str, required=True, help="Path to model config YAML")
	parser.add_argument("--data", type=str, required=True, help="Path to preprocessed crops directory (must contain labels.csv)")
	parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
	parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
	parser.add_argument("--threshold", type=float, default=15.0, help="Eyes-off-road threshold angle in degrees")
	parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
	args = parser.parse_args()

	# configure logging
	level = logging.DEBUG if args.verbose else logging.INFO
	logging.basicConfig(
		level=level,
		format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
	)

	# load config
	try:
		config = Config.from_path(args.config)
	except (FileNotFoundError, MalformedConfig) as e:
		logger.error(e)
		return

	logger.info(f"Loaded config: {config.name}")

	# load model
	model = get_model(config)
	model.load()
	logger.info("Model loaded")

	# warmup
	model.warmup(args.warmup)
	logger.info(f"Warmup complete ({args.warmup} iterations)")

	# load data
	labels_csv = str(Path(args.data) / "labels.csv")
	filenames, y_true = load_labels(labels_csv)
	logger.info(f"Loaded {len(filenames)} samples")

	# run inference in batches
	batch_size = args.batch_size
	logger.info(f"Running inference (batch_size={batch_size})...")

	transform = transforms.Compose([
		transforms.Resize(config.input_resolution),
		transforms.ConvertImageDtype(torch.float32),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	all_predictions = []
	with torch.no_grad():
		for i in range(0, len(filenames), batch_size):
			batch_filenames = filenames[i:i + batch_size]
			batch_images = []
			for fname in batch_filenames:
				img = tv_io.read_image(str(Path(args.data) / fname))
				img = transform(img)
				batch_images.append(img)

			batch_tensor = torch.stack(batch_images)
			preds = model.infer(batch_tensor)
			all_predictions.append(preds.cpu())

			if (i // batch_size) % 10 == 0:
				logger.info(f"  Batch {i // batch_size + 1}/{(len(filenames) + batch_size - 1) // batch_size}")

	y_pred = torch.cat(all_predictions).numpy()
	logger.info("Inference complete")

	# generate report
	result = report(y_pred, y_true, threshold=args.threshold)

	# log to MLflow
	mlflow.set_experiment("gaze-eval")
	with mlflow.start_run(run_name=config.name):
		# params
		mlflow.log_param("model_name", config.name)
		mlflow.log_param("weights", config.weights)
		mlflow.log_param("runtime", config.runtime)
		mlflow.log_param("quantization", config.quantization)
		mlflow.log_param("input_resolution", config.input_resolution)
		mlflow.log_param("batch_size", args.batch_size)
		mlflow.log_param("threshold_angle", args.threshold)
		mlflow.log_param("num_samples", len(filenames))

		# accuracy metrics
		mlflow.log_metric("mae_yaw", result.accuracy.mae_yaw)
		mlflow.log_metric("mae_pitch", result.accuracy.mae_pitch)
		mlflow.log_metric("mae_combined", result.accuracy.mae_combined)
		mlflow.log_metric("p50_yaw", result.accuracy.percentile_errors_yaw[0])
		mlflow.log_metric("p95_yaw", result.accuracy.percentile_errors_yaw[1])
		mlflow.log_metric("p99_yaw", result.accuracy.percentile_errors_yaw[2])
		mlflow.log_metric("p50_pitch", result.accuracy.percentile_errors_pitch[0])
		mlflow.log_metric("p95_pitch", result.accuracy.percentile_errors_pitch[1])
		mlflow.log_metric("p99_pitch", result.accuracy.percentile_errors_pitch[2])

		# safety metrics
		mlflow.log_metric("fpr", result.safety.fpr)
		mlflow.log_metric("fnr", result.safety.fnr)

		# slicing metrics
		mlflow.log_metric("mae_slice_0_15", result.slicing.zero_to_fifteen.mae_combined)
		mlflow.log_metric("mae_slice_15_30", result.slicing.fifteen_to_thirty.mae_combined)
		mlflow.log_metric("mae_slice_30_45", result.slicing.thirty_to_forty_five.mae_combined)
		mlflow.log_metric("mae_slice_45_plus", result.slicing.forty_five_plus.mae_combined)

		# log the config file as artifact
		mlflow.log_artifact(args.config)

	logger.info("Results logged to MLflow")

	# print results
	print("\n" + "=" * 60)
	print(f"Evaluation Report: {config.name}")
	print("=" * 60)
	print(f"\nAccuracy:")
	print(f"  MAE (yaw):      {result.accuracy.mae_yaw:.2f}°")
	print(f"  MAE (pitch):    {result.accuracy.mae_pitch:.2f}°")
	print(f"  MAE (combined): {result.accuracy.mae_combined:.2f}°")
	print(f"  p50/p95/p99 yaw:   {result.accuracy.percentile_errors_yaw}")
	print(f"  p50/p95/p99 pitch: {result.accuracy.percentile_errors_pitch}")
	print(f"\nSafety (threshold={args.threshold}°):")
	print(f"  FPR: {result.safety.fpr:.4f}")
	print(f"  FNR: {result.safety.fnr:.4f}")
	print(f"\nSlicing by yaw bin:")
	print(f"  0-15°:  MAE = {result.slicing.zero_to_fifteen.mae_combined:.2f}°")
	print(f"  15-30°: MAE = {result.slicing.fifteen_to_thirty.mae_combined:.2f}°")
	print(f"  30-45°: MAE = {result.slicing.thirty_to_forty_five.mae_combined:.2f}°")
	print(f"  45+°:   MAE = {result.slicing.forty_five_plus.mae_combined:.2f}°")
	print("=" * 60)

	# teardown
	model.teardown()
	logger.info("Teardown complete")


if __name__ == "__main__":
	main()

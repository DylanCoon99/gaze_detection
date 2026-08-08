from typing import Any, Dict
from dataclasses import dataclass, field
import yaml
import cv2
import mediapipe as mp
import scipy.io as sio 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from PIL import Image
import numpy as np
from config import PipelineConfig
import logging
import csv
import os

logger = logging.getLogger("preprocessing")



@dataclass
class Detection:
	x: float
	y: float
	width: float
	height: float


@dataclass
class Label:
	pitch: float
	yaw: float




class Pipeline:

	def __init__(self, config: PipelineConfig):
		# initialize the config
		self.config = config

		# initialize self.detector
		base_options = python.BaseOptions(model_asset_path=self.config.detection_model)
		options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=self.config.confidence)
		self.detector = vision.FaceDetector.create_from_options(options)

	def detect(self, img_file: Path):
		'''
		takes a raw image, runs a face detector (like MediaPipe BlazeFace), and 
		returns a bounding box: the x, y, width, height of where the face is in the
		image. 
		'''

		# I am wondering if I need to resize the image before performing inference

		# face detector: MediaPipe BlazeFace
		image = mp.Image.create_from_file(str(img_file))

		detection_result = self.detector.detect(image)

		if detection_result.detections:
			x = detection_result.detections[0].bounding_box.origin_x
			y = detection_result.detections[0].bounding_box.origin_y
			width = detection_result.detections[0].bounding_box.width
			height = detection_result.detections[0].bounding_box.height

			return Detection(x=x, y=y, width=width, height=height)

		return None


	def crop(self, img_file, detection: Detection):
		'''
		Takes the image and the bounding box from detect, expands the box by the margin
		percentage, clips it to stay within image bounds, cuts out that region, and
		resizes it to the target resolution (e.g. 224x224). Returns the cropped face image.
		'''

		# open the image
		image = Image.open(img_file)
		img_width, img_height = image.size

		# expand the bounding box by margin on each side
		margin_w = detection.width * self.config.margin
		margin_h = detection.height * self.config.margin

		x1 = detection.x - margin_w
		y1 = detection.y - margin_h
		x2 = detection.x + detection.width + margin_w
		y2 = detection.y + detection.height + margin_h

		# clip to image bounds
		x1 = max(0, x1)
		y1 = max(0, y1)
		x2 = min(img_width, x2)
		y2 = min(img_height, y2)

		# crop and resize
		cropped = image.crop((x1, y1, x2, y2))
		H, W = self.config.input_resolution
		cropped = cropped.resize((W, H))

		return cropped


	def labels(self, mat_file) -> Label:
		'''
		reads the .mat file for a given image and extracts yaw and pitch in degrees. 
		I already wrote this logic in my day 6 dataset class. This just isolates it 
		so the pipeline can save the labels alongside the crops.
		'''

		mat = sio.loadmat(mat_file)
		pose = mat['Pose_Para'][0]
		pitch, yaw = pose[0], pose[1]
		pitch = pitch * 180 / np.pi
		yaw = yaw * 180 / np.pi

		return Label(pitch=pitch, yaw=yaw)


	def run(self):
		'''
		The pipeline orchestrates them: for each image in the input directory, run
		detect → crop → labels → save the crop and its label to the output directory.
		'''

		input_path = Path(self.config.input_directory)
		output_path = Path(self.config.output_directory)
		labels_path = output_path / "labels.csv"

		mat_files = sorted(input_path.glob("*.mat"))
		logger.info(f"Found {len(mat_files)} .mat files in {input_path}")

		processed = 0
		skipped_no_image = 0
		skipped_no_face = 0

		with open(labels_path, mode="w", newline="", encoding="utf-8") as file:
			writer = csv.DictWriter(file, fieldnames=["filename", "yaw", "pitch"])
			writer.writeheader()

			for mat_file in mat_files:

				# find corresponding image file
				img_file = mat_file.with_suffix(".jpg")
				if not img_file.exists():
					skipped_no_image += 1
					logger.debug(f"No image found for {mat_file.name}, skipping")
					continue

				# perform every step in the pipeline
				detection = self.detect(img_file)
				if not detection:
					skipped_no_face += 1
					logger.warning(f"No face detected in {img_file.name}, skipping")
					continue

				cropped_img = self.crop(img_file, detection)
				label = self.labels(mat_file)

				logger.debug(f"Detection: {detection}")
				logger.debug(f"Label: {label}")

				cropped_img.save(output_path / img_file.name)
				writer.writerow({"filename": img_file.name, "yaw": label.yaw, "pitch": label.pitch})

				processed += 1
				if processed % 100 == 0:
					logger.info(f"Processed {processed} images...")

		logger.info(f"Pipeline complete: {processed} processed, {skipped_no_image} missing images, {skipped_no_face} no face detected")

		return



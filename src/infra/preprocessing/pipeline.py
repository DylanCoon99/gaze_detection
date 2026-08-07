from typing import Any, Dict
from dataclasses import dataclass, field
import yaml
import cv2
import mediapipe as mp
from pathlib import Path


'''

Config stores this:
	(input directory, output directory, target resolution, etc.)


'''


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



# Define the config here

@dataclass(frozen=True)
class PipelineConfig:
	input_directory: str
	output_directory: str
	margin: float
	input_resolution: list[int] = field(default_factory=list)


	@classmethod
	def from_path(cls, yaml_path: str):

		try:
			with open(yaml_path, "r") as file:
				# loads the yaml to a dictionary
				config = yaml.safe_load(file)

		except FileNotFoundError:
			raise FileNotFoundError(f"File: {yaml_path} not found!")

		# validate the yaml is the correct format

		return cls.from_dict(config)



	@classmethod
	def from_dict(cls, yaml_dict):
		try:

			# validate the yaml
			# need to check if the values are none

			input_directory  = yaml_dict["input_directory"]
			output_directory = yaml_dict["output_directory"]
			margin           = yaml_dict["margin"]
			input_resolution = yaml_dict["input_resolution"]

			return cls(
				input_directory=input_directory, 
				output_directory=output_directory,
				margin=margin,
				input_resolution=input_resolution
			)
		
		except KeyError as e:
			raise MalformedConfig(error_msg=e)




class Pipeline:

	def __init__(self, config: PipelineConfig):
		# initialize the config
		self.config = config
		self.mp_face_detection = mp.solutions.face_detection

	def detect(self, image):
		'''
		takes a raw image, runs a face detector (like MediaPipe BlazeFace), and 
		returns a bounding box: the x, y, width, height of where the face is in the
		image. 
		'''

		# pick a face detector


		# need to determine if detect can return multiple detections per image



		return


	def crop(self, detection: Detection):
		'''
		takes the image and the bounding box from detect, expands the box by the margin 
		percentage, clips it to stay within image bounds, cuts out that region, and 
		resizes it to the target resolution (e.g. 224x224). Returns the cropped face image.
		'''

		return


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

		path = Path(self.config.input_directory)

		# iterate over all labels in the input directory
		for mat_file in sorted(path.glob("*.mat")):

			# find corresponding image file
            img_file = mat_file.with_suffix(".jpg")
            if not img_file.exists():
            	# if image file does not exist -> skip this .mat file
                continue
            
            # perform every step in the pipeline
			detection = self.detect(img_file)
			cropped_img = self.crop(detection)
   			label = self.labels(mat_file)

            # TODO: save the crop and the label to the output directory
            output_directory = self.config.output_directory
            



		return



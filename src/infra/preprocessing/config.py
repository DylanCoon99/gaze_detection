from dataclasses import dataclass, field
import yaml


# Define the pipeline config here
@dataclass(frozen=True)
class PipelineConfig:
	input_directory: str
	output_directory: str
	margin: float
	confidence: float
	detection_model: str
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
			confidence       = yaml_dict["confidence"]
			detection_model  = yaml_dict["detection_model"]
			input_resolution = yaml_dict["input_resolution"]

			return cls(
				input_directory=input_directory, 
				output_directory=output_directory,
				margin=margin,
				confidence=confidence,
				detection_model=detection_model,
				input_resolution=input_resolution
			)
		
		except KeyError as e:
			# this might raise an error; I haven't defined MalformedConfig in this scope
			raise MalformedConfig(error_msg=e)


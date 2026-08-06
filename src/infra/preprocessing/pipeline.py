from typing import Any, Dict


'''

Config stores this:
	(input directory, output directory, target resolution, etc.)


'''


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


	def crop(self):

		return


	def detect(self):

		return


	def labels(self):

		return


	def run(self):

		# perform every step in the pipeline
		self.crop()
		self.detect()
		self.labels()

		return



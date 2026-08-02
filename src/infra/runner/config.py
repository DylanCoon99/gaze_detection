from dataclasses import dataclass, field
from errors import MalformedConfig
import yaml


'''
name: mobilenet_v2_fp32
weights: models/mnv2_fp32.pth
runtime: torch
input_resolution: [224, 224]
quantization: none
preprocessing: standard_crop_v2 
'''



# define a config object


@dataclass(frozen=True)
class Config:
	name: str
	weights: str
	runtime: str
	quantization: str
	preprocessing: str
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

			name             = yaml_dict["name"]
			weights          = yaml_dict["weights"]
			runtime          = yaml_dict["runtime"]
			quantization     = yaml_dict["quantization"]
			preprocessing    = yaml_dict["preprocessing"]
			input_resolution = yaml_dict["input_resolution"]

			return cls(
				name=name, 
				weights=weights,
				runtime=runtime,
				quantization=quantization,
				preprocessing=preprocessing,
				input_resolution=input_resolution
			)
		
		except KeyError as e:
			raise MalformedConfig(error_msg=e)


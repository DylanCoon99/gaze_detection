from dataclasses import dataclass, field
from errors import MalformedConfig


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
	input_resolution: list[str] = field(default_factory=list)
	quantization: str
	preprocessing: str


    @classmethod
    def from_path(cls, yaml_path: str):

    	with open(yaml_path, "r") as file:
    		# loads the yaml to a dictionary
		    config = yaml.safe_load(file)

		# validate the yaml is the correct format
		try:
			name             = config["name"]
			weights          = config["weights"]
			runtime          = config["runtime"]
			# I am wondering if the types are inferred when the yaml is load; this is supposed to be a list
			# I hope it is not a string; need to check
			input_resolution = config["input_resolution"]
			quantization     = config["quantization"]
			preprocessing    = config["preprocessing"]

			return cls(
				name=name, 
				weights=weights,
				runtime=runtime,
				input_resolution=input_resolution,
				quantization=quantization,
				preprocessing=preprocessing
			)
		
		except Exception as e:
			raise MalformedConfig(error_msg=e)


    @classmethod
    def from_dict(cls, yaml_dict):
        try:
			name             = yaml_dict["name"]
			weights          = yaml_dict["weights"]
			runtime          = yaml_dict["runtime"]
			input_resolution = yaml_dict["input_resolution"]
			quantization     = yaml_dict["quantization"]
			preprocessing    = yaml_dict["preprocessing"]

			return cls(
				name=name, 
				weights=weights,
				runtime=runtime,
				input_resolution=input_resolution,
				quantization=quantization,
				preprocessing=preprocessing
			)
		
		except Exception as e:
			raise MalformedConfig(error_msg=e)


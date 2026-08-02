# Model Evaluation Runner Implementation
from config import Config
from errors import MalformedConfig


# Testing the config yaml parsing


def main():

	file_path = "/Users/Dylan/Documents/gaze_detection/src/infra/runner/test_config_in.yaml"


	try:
		config = Config.from_path(file_path)
	except FileNotFoundError as e:
		print(e)
		return
	except MalformedConfig as e:
		print(e)
		return


	# should be able to safely access keys at this point
	name             = config.name
	weights          = config.weights
	runtime          = config.runtime
	input_resolution = config.input_resolution
	quantization     = config.quantization
	preprocessing    = config.preprocessing

	print(f"name: {name}")
	print(f"weights: {weights}")
	print(f"runtime: {runtime}")
	print(f"input_resolution: {input_resolution}")
	print(f"quantization: {quantization}")
	print(f"preprocessing: {preprocessing}")

	return





if __name__ == "__main__":
	main()
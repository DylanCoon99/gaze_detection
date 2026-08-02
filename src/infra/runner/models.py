
from abc import ABC, abstractmethod
from config import Config


'''

name: mobilenet_v2_fp32

weights: models/mnv2_fp32.pth
runtime: torch
input_resolution: [224, 224]
quantization: none
preprocessing: standard_crop_v2 

'''


def get_model(config: Config):

	# the config determines which model we instantiate

	models = {
		"torch": TorchModel,
		# can add more for each runtime
	}
	model = models.get(config.runtime)
	if model:
		return model()
	raise ValueError(f"Unknown runtime {config.runtime}")

	

class BaseModel(ABC):

	def __init__(self, config: Config):
		self.config = config
		return


	@abstractmethod
	def load(self):
		pass

	@abstractmethod
	def warmup(self, n: int):
		# perform n inferences
		# warms up the CPU cache
		# first inferences can be slow so we throw these away
		pass

	@abstractmethod
	def infer(self, batch):
		pass

	@abstractmethod
	def teardown(self):
		pass



# child classes

class TorchModel(BaseModel):

	def load(self):
		# configure the model based on self.config
		

		return

	def warmup(self, n: int):
		# perform n inferences
		# warms up the CPU cache
		# first inferences can be slow so we throw these away


		return

	def infer(self, batch):

		return

	def teardown(self):

		return



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

	def load():

		return

	def warmup(n: int):

		return

	def infer(batch):

		return

	def teardown():

		return


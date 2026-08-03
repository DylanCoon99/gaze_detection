from abc import ABC, abstractmethod
from config import Config
import torch

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
		self.model = None
		self.transform = None
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
		# configure the model with pytorch runtime

		# MobileNetV2 FP32, ResNet-18 FP32, MobileViT-small

		# load the weights
		self.model = torch.jit.load(self.config.weights)

		self.model.eval()

		return

	def warmup(self, n: int):
		# perform n inferences
		# warms up the CPU cache
		# first inferences can be slow so we throw these away


		return

	def infer(self, batch):
		# perform inference on the batch

		return

	def teardown(self):



		return


from abc import ABC, abstractmethod

import torch

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
		self.model = None
		self.transform = None
		self.device = None
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


		# think about this later
		'''
		# for running on mac
		if torch.backends.mps.is_available():
		    self.device = torch.device("mps")
		    print("MPS device found!")
		else:
		    self.device = torch.device("cpu")
		    print("MPS not available. Using CPU.")

		self.model.to(self.device)
		'''

		return

	def warmup(self, n: int):
		# perform n inferences
		# warms up the CPU cache
		# first inferences can be slow so we throw these away

		# saves time
		with torch.no_grad():

			# create a dummy tensor of size (1, 3, H, W)
			H, W = self.config.input_resolution
			dummy_tensor = torch.rand(1, 3, H, W)

			dummy_tensor = dummy_tensor.to(self.device)

			# perform inference on the tensor n times
			for i in range(n):
				self.model(dummy_tensor)

		return

	def infer(self, batch):
		# perform inference on the batch
		with torch.no_grad():

			batch = batch.to(self.device)

			return self.model(batch)

	def teardown(self):

		self.model = None
		
		return


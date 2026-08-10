from config import Config
from models import TorchModel


def get_model(config: Config):
	"""Factory: maps config runtime to the correct model subclass."""

	models = {
		"torch": TorchModel,
		# can add more for each runtime
	}
	model = models.get(config.runtime)
	if model:
		return model(config)
	raise ValueError(f"Unknown runtime: {config.runtime}")

def get_model(config: Config):

	# the config determines which model we instantiate

	models = {
		"torch": TorchModel,
		# can add more for each runtime
	}
	model = models.get(config.runtime)
	if model:
		return model(config)
	raise ValueError(f"Unknown runtime {config.runtime}")
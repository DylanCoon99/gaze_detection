# Custom Exceptions


class MalformedConfig(Exception):
	def __init__(self, error_msg="check yaml"):
		# Pass a formatted message to the base Exception class
		super().__init__(f"Malformed config file: {error_msg}")
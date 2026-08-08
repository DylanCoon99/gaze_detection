# End-to-end pipeline — runs detect + crop over a dataset directory and saves output
import argparse
import logging

from config import PipelineConfig
from pipeline import Pipeline


logger = logging.getLogger("preprocessing")


def main():

	parser = argparse.ArgumentParser(description="Run the preprocessing pipeline")
	parser.add_argument("--config", type=str, default="test_pipeline_config.yaml", help="Path to pipeline config YAML")
	parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
	args = parser.parse_args()

	# configure logging
	level = logging.DEBUG if args.verbose else logging.WARNING
	logging.basicConfig(
		level=level,
		format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
	)

	config = PipelineConfig.from_path(args.config)

	my_pipeline = Pipeline(config)

	my_pipeline.run()


	return



if __name__ == "__main__":
	main()
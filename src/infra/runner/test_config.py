import pytest
import os
from config import Config
from errors import MalformedConfig

# Test with: pytest test_config.py -v

# ── Fixtures ──

VALID_DICT = {
    "name": "mobilenet_v2_fp32",
    "weights": "models/mnv2_fp32.pth",
    "runtime": "torch",
    "input_resolution": [224, 224],
    "quantization": "none",
    "preprocessing": "standard_crop_v2",
}

TEST_DIR = os.path.dirname(__file__)
VALID_YAML = os.path.join(TEST_DIR, "test_config.yaml")
INVALID_YAML = os.path.join(TEST_DIR, "test_config_invalid.yaml")


# ── from_dict tests ──

class TestFromDict:

    def test_valid_config(self):
        config = Config.from_dict(VALID_DICT)
        assert config.name == "mobilenet_v2_fp32"
        assert config.weights == "models/mnv2_fp32.pth"
        assert config.runtime == "torch"
        assert config.input_resolution == [224, 224]
        assert config.quantization == "none"
        assert config.preprocessing == "standard_crop_v2"

    def test_missing_key_raises_malformed(self):
        incomplete = {"name": "test", "weights": "test.pth"}
        with pytest.raises(MalformedConfig):
            Config.from_dict(incomplete)

    def test_empty_dict_raises_malformed(self):
        with pytest.raises(MalformedConfig):
            Config.from_dict({})

    def test_config_is_frozen(self):
        config = Config.from_dict(VALID_DICT)
        with pytest.raises(AttributeError):
            config.name = "something_else"


# ── from_path tests ──

class TestFromPath:

    def test_valid_yaml(self):
        config = Config.from_path(VALID_YAML)
        assert config.name == "mobilenet_v2_fp32"
        assert config.runtime == "torch"
        assert config.input_resolution == [224, 224]
        assert config.quantization == "none"

    def test_invalid_yaml_raises_malformed(self):
        with pytest.raises(MalformedConfig):
            Config.from_path(INVALID_YAML)

    def test_missing_file_raises_not_found(self):
        with pytest.raises(FileNotFoundError):
            Config.from_path("/nonexistent/path/config.yaml")

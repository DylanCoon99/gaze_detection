import pytest
import os
import torch
import torch.nn as nn
from config import Config
from models import TorchModel

# Test with: pytest test_models.py -v

TEST_DIR = os.path.dirname(__file__)
DUMMY_MODEL_PATH = os.path.join(TEST_DIR, "test_dummy_model.pt")


# ── Helper: create a tiny TorchScript model for testing ──

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3 * 32 * 32, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)


@pytest.fixture(autouse=True)
def dummy_model_file():
    """Create a tiny TorchScript model file before tests, clean up after."""
    model = TinyModel()
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, DUMMY_MODEL_PATH)
    yield DUMMY_MODEL_PATH
    if os.path.exists(DUMMY_MODEL_PATH):
        os.remove(DUMMY_MODEL_PATH)


@pytest.fixture
def torch_config():
    return Config(
        name="test_model",
        weights=DUMMY_MODEL_PATH,
        runtime="torch",
        quantization="none",
        preprocessing="standard_crop_v2",
        input_resolution=[32, 32],
    )


# ── load tests ──

class TestLoad:

    def test_load_sets_model(self, torch_config):
        m = TorchModel(torch_config)
        assert m.model is None
        m.load()
        assert m.model is not None

    def test_load_invalid_weights_raises(self):
        config = Config(
            name="bad_model",
            weights="/nonexistent/model.pt",
            runtime="torch",
            quantization="none",
            preprocessing="standard_crop_v2",
            input_resolution=[32, 32],
        )
        m = TorchModel(config)
        with pytest.raises(Exception):
            m.load()


# ── warmup tests ──

class TestWarmup:

    def test_warmup_runs_without_error(self, torch_config):
        m = TorchModel(torch_config)
        m.load()
        m.warmup(3)

    def test_warmup_zero_iterations(self, torch_config):
        m = TorchModel(torch_config)
        m.load()
        m.warmup(0)


# ── infer tests ──

class TestInfer:

    def test_infer_returns_output(self, torch_config):
        m = TorchModel(torch_config)
        m.load()
        batch = torch.rand(1, 3, 32, 32)
        output = m.infer(batch)
        assert output is not None

    def test_infer_output_shape(self, torch_config):
        m = TorchModel(torch_config)
        m.load()
        batch = torch.rand(4, 3, 32, 32)
        output = m.infer(batch)
        assert output.shape == (4, 2)

    def test_infer_before_load_raises(self, torch_config):
        m = TorchModel(torch_config)
        batch = torch.rand(1, 3, 32, 32)
        with pytest.raises(Exception):
            m.infer(batch)


# ── teardown tests ──

class TestTeardown:

    def test_teardown_clears_model(self, torch_config):
        m = TorchModel(torch_config)
        m.load()
        assert m.model is not None
        m.teardown()
        assert m.model is None

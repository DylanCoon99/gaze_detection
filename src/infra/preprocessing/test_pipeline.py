import pytest
import csv
import numpy as np
import scipy.io as sio
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock
from pipeline import Pipeline, Detection, Label
from config import PipelineConfig

# Test with: cd src/infra/preprocessing && pytest test_pipeline.py -v


# ── Fixtures ──

@pytest.fixture
def tmp_dirs(tmp_path):
    """Create input and output directories."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    return input_dir, output_dir


@pytest.fixture
def sample_image(tmp_dirs):
    """Create a 450x450 test image with a face-like region in the center."""
    input_dir, _ = tmp_dirs
    img = Image.new("RGB", (450, 450), color=(200, 180, 160))
    img_path = input_dir / "test_001.jpg"
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_mat(tmp_dirs):
    """Create a .mat file with pose parameters."""
    input_dir, _ = tmp_dirs
    mat_path = input_dir / "test_001.mat"
    # pitch=0.5 rad (~28.6°), yaw=-0.3 rad (~-17.2°)
    pose_para = np.array([[0.5, -0.3, 0.1, 0, 0, 0, 1.0]])
    sio.savemat(str(mat_path), {"Pose_Para": pose_para})
    return mat_path


@pytest.fixture
def pipeline_config(tmp_dirs):
    """Create a PipelineConfig pointing to tmp dirs."""
    input_dir, output_dir = tmp_dirs
    return PipelineConfig(
        input_directory=str(input_dir),
        output_directory=str(output_dir),
        margin=0.2,
        confidence=0.5,
        detection_model="blaze_face_short_range.tflite",
        input_resolution=[224, 224],
    )


@pytest.fixture
def mock_pipeline(pipeline_config):
    """Create a Pipeline with a mocked MediaPipe detector."""
    with patch("pipeline.python.BaseOptions"), \
         patch("pipeline.vision.FaceDetectorOptions"), \
         patch("pipeline.vision.FaceDetector.create_from_options") as mock_create:
        mock_detector = MagicMock()
        mock_create.return_value = mock_detector
        p = Pipeline(pipeline_config)
        p.detector = mock_detector
        return p


# ── crop tests ──

class TestCrop:

    def test_crop_returns_correct_size(self, mock_pipeline, sample_image):
        detection = Detection(x=100, y=100, width=200, height=200)
        result = mock_pipeline.crop(sample_image, detection)
        assert result.size == (224, 224)

    def test_crop_clips_to_image_bounds(self, mock_pipeline, sample_image):
        # detection near edge — margin would push past 0
        detection = Detection(x=5, y=5, width=100, height=100)
        result = mock_pipeline.crop(sample_image, detection)
        assert result.size == (224, 224)

    def test_crop_clips_bottom_right(self, mock_pipeline, sample_image):
        # detection near bottom-right — margin would push past 450
        detection = Detection(x=350, y=350, width=100, height=100)
        result = mock_pipeline.crop(sample_image, detection)
        assert result.size == (224, 224)


# ── labels tests ──

class TestLabels:

    def test_labels_extracts_yaw_pitch(self, mock_pipeline, sample_mat):
        label = mock_pipeline.labels(sample_mat)
        assert isinstance(label, Label)
        assert pytest.approx(label.pitch, abs=0.1) == 28.6
        assert pytest.approx(label.yaw, abs=0.1) == -17.2

    def test_labels_returns_degrees(self, mock_pipeline, sample_mat):
        label = mock_pipeline.labels(sample_mat)
        # values should be in degrees, not radians
        assert abs(label.pitch) > 1.0
        assert abs(label.yaw) > 1.0


# ── run tests ──

class TestRun:

    def _make_mock_detection_result(self, x, y, w, h):
        """Helper to create a mock MediaPipe detection result."""
        mock_bbox = MagicMock()
        mock_bbox.origin_x = x
        mock_bbox.origin_y = y
        mock_bbox.width = w
        mock_bbox.height = h
        mock_detection = MagicMock()
        mock_detection.bounding_box = mock_bbox
        mock_result = MagicMock()
        mock_result.detections = [mock_detection]
        return mock_result

    def test_run_produces_output_files(self, mock_pipeline, sample_image, sample_mat, tmp_dirs):
        _, output_dir = tmp_dirs
        mock_pipeline.detector.detect.return_value = self._make_mock_detection_result(100, 100, 200, 200)

        mock_pipeline.run()

        # check cropped image was saved
        assert (output_dir / "test_001.jpg").exists()

        # check labels.csv was created with correct content
        labels_path = output_dir / "labels.csv"
        assert labels_path.exists()
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["filename"] == "test_001.jpg"
            assert float(rows[0]["yaw"]) == pytest.approx(-17.2, abs=0.1)

    def test_run_skips_missing_image(self, mock_pipeline, tmp_dirs):
        input_dir, output_dir = tmp_dirs
        # create a .mat with no matching .jpg
        mat_path = input_dir / "orphan.mat"
        pose_para = np.array([[0.5, -0.3, 0.1, 0, 0, 0, 1.0]])
        sio.savemat(str(mat_path), {"Pose_Para": pose_para})

        mock_pipeline.run()

        # labels.csv should exist but have only the header
        labels_path = output_dir / "labels.csv"
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 0

    def test_run_skips_no_face_detected(self, mock_pipeline, sample_image, sample_mat, tmp_dirs):
        _, output_dir = tmp_dirs
        # detector returns no detections
        mock_result = MagicMock()
        mock_result.detections = []
        mock_pipeline.detector.detect.return_value = mock_result

        mock_pipeline.run()

        # no cropped image should be saved
        assert not (output_dir / "test_001.jpg").exists()

        # labels.csv should have only header
        labels_path = output_dir / "labels.csv"
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 0

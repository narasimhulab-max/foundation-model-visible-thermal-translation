from src.config import (
    PROJECT_ROOT,
    MODELS_DIR,
    OUTPUTS_DIR,
    SUPPORTED_IMAGE_FORMATS,
)


def test_project_root_exists():
    assert PROJECT_ROOT.exists()


def test_models_directory_path():
    assert MODELS_DIR.name == "models"


def test_outputs_directory_path():
    assert OUTPUTS_DIR.name == "outputs"


def test_supported_image_formats():
    assert ".png" in SUPPORTED_IMAGE_FORMATS
    assert ".jpg" in SUPPORTED_IMAGE_FORMATS
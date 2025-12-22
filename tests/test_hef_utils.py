import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from hailo_apps.python.core.common.hef_utils import get_hef_labels_json
from hailo_apps.python.core.common.defines import RESOURCES_JSON_DIR_NAME

@patch("hailo_apps.python.core.common.hef_utils.get_resource_path")
def test_get_hef_labels_json(mock_get_resource_path):
    # Setup mock return value
    mock_get_resource_path.side_effect = lambda pipeline_name, resource_type, arch, model: Path(f"/path/to/resources/json/{model}")

    # Test hailo_4_classes
    # Should match "hailo_4_classes" inside the string
    result = get_hef_labels_json("path/to/hailo_yolov8n_4_classes_vga.hef")
    assert str(result) == "/path/to/resources/json/hailo_4_classes.json"
    mock_get_resource_path.assert_called_with(
        pipeline_name=None,
        resource_type=RESOURCES_JSON_DIR_NAME,
        arch=None,
        model="hailo_4_classes.json"
    )

    # Test barcode
    # Should match "barcode"
    result = get_hef_labels_json("/data/yolov8s-hailo8-barcode.hef")
    assert str(result) == "/path/to/resources/json/barcode_labels.json"
    mock_get_resource_path.assert_called_with(
        pipeline_name=None,
        resource_type=RESOURCES_JSON_DIR_NAME,
        arch=None,
        model="barcode_labels.json"
    )

    # Test visdrone
    # Should match "visdrone"
    result = get_hef_labels_json("ssd_mobilenet_v1_visdrone.hef")
    assert str(result) == "/path/to/resources/json/visdrone.json"
    mock_get_resource_path.assert_called_with(
        pipeline_name=None,
        resource_type=RESOURCES_JSON_DIR_NAME,
        arch=None,
        model="visdrone.json"
    )

    # Test random hef
    result = get_hef_labels_json("random_model.hef")
    assert result is None


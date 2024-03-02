from unittest.mock import patch

from swarms.models.ultralytics_model import UltralyticsModel


def test_ultralytics_init():
    with patch("swarms.models.YOLO") as mock_yolo:
        model_name = "yolov5s"
        ultralytics = UltralyticsModel(model_name)
        mock_yolo.assert_called_once_with(model_name)
        assert ultralytics.model_name == model_name
        assert ultralytics.model == mock_yolo.return_value


def test_ultralytics_call():
    with patch("swarms.models.YOLO") as mock_yolo:
        model_name = "yolov5s"
        ultralytics = UltralyticsModel(model_name)
        task = "detect"
        args = (1, 2, 3)
        kwargs = {"a": "A", "b": "B"}
        result = ultralytics(task, *args, **kwargs)
        mock_yolo.return_value.assert_called_once_with(
            task, *args, **kwargs
        )
        assert result == mock_yolo.return_value.return_value


def test_ultralytics_list_models():
    with patch("swarms.models.YOLO") as mock_yolo:
        model_name = "yolov5s"
        ultralytics = UltralyticsModel(model_name)
        result = ultralytics.list_models()
        mock_yolo.list_models.assert_called_once()
        assert result == mock_yolo.list_models.return_value

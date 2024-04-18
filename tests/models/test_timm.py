from unittest.mock import patch

import torch

from swarms.models import TimmModel


def test_timm_model_init():
    with patch("swarms.models.timm.list_models") as mock_list_models:
        model_name = "resnet18"
        pretrained = True
        in_chans = 3
        timm_model = TimmModel(model_name, pretrained, in_chans)
        mock_list_models.assert_called_once()
        assert timm_model.model_name == model_name
        assert timm_model.pretrained == pretrained
        assert timm_model.in_chans == in_chans
        assert timm_model.models == mock_list_models.return_value


def test_timm_model_call():
    with patch("swarms.models.timm.create_model") as mock_create_model:
        model_name = "resnet18"
        pretrained = True
        in_chans = 3
        timm_model = TimmModel(model_name, pretrained, in_chans)
        task = torch.rand(1, in_chans, 224, 224)
        result = timm_model(task)
        mock_create_model.assert_called_once_with(
            model_name, pretrained=pretrained, in_chans=in_chans
        )
        assert result == mock_create_model.return_value(task)


def test_timm_model_list_models():
    with patch("swarms.models.timm.list_models") as mock_list_models:
        model_name = "resnet18"
        pretrained = True
        in_chans = 3
        timm_model = TimmModel(model_name, pretrained, in_chans)
        result = timm_model.list_models()
        mock_list_models.assert_called_once()
        assert result == mock_list_models.return_value

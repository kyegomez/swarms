import pytest
import torch
from torch import nn
from swarms.utils import load_model_torch


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


# Test case 1: Test if model can be loaded successfully
def test_load_model_torch_success(tmp_path):
    model = DummyModel()
    # Save the model to a temporary directory
    model_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_path)

    # Load the model
    model_loaded = load_model_torch(model_path, model=DummyModel())

    # Check if loaded model has the same architecture
    assert isinstance(
        model_loaded, DummyModel
    ), "Loaded model type mismatch."


# Test case 2: Test if function raises FileNotFoundError for non-existent file
def test_load_model_torch_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_model_torch("non_existent_model.pt")


# Test case 3: Test if function catches and raises RuntimeError for invalid model file
def test_load_model_torch_invalid_file(tmp_path):
    file = tmp_path / "invalid_model.pt"
    file.write_text("Invalid model file.")

    with pytest.raises(RuntimeError):
        load_model_torch(file)


# Test case 4: Test for handling of 'strict' parameter
def test_load_model_torch_strict_handling(tmp_path):
    # Create a model and modify it to cause a mismatch
    model = DummyModel()
    model.fc = nn.Linear(10, 3)
    model_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_path)

    # Try to load the modified model with 'strict' parameter set to True
    with pytest.raises(RuntimeError):
        load_model_torch(model_path, model=DummyModel(), strict=True)


# Test case 5: Test for 'device' parameter handling
def test_load_model_torch_device_handling(tmp_path):
    model = DummyModel()
    model_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_path)

    # Define a device other than default and load the model to the specified device
    device = torch.device("cpu")
    model_loaded = load_model_torch(
        model_path, model=DummyModel(), device=device
    )

    assert (
        model_loaded.fc.weight.device == device
    ), "Model not loaded to specified device."


# Test case 6: Testing for correct handling of '*args' and '**kwargs'
def test_load_model_torch_args_kwargs_handling(monkeypatch, tmp_path):
    model = DummyModel()
    model_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_path)

    def mock_torch_load(*args, **kwargs):
        assert (
            "pickle_module" in kwargs
        ), "Keyword arguments not passed to 'torch.load'."

    # Monkeypatch 'torch.load' to check if '*args' and '**kwargs' are passed correctly
    monkeypatch.setattr(torch, "load", mock_torch_load)
    load_model_torch(
        model_path, model=DummyModel(), pickle_module="dummy_module"
    )


# Test case 7: Test for model loading on CPU if no GPU is available
def test_load_model_torch_cpu(tmp_path):
    model = DummyModel()
    model_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_path)

    def mock_torch_cuda_is_available():
        return False

    # Monkeypatch to simulate no GPU available
    pytest.MonkeyPatch.setattr(
        torch.cuda, "is_available", mock_torch_cuda_is_available
    )
    model_loaded = load_model_torch(model_path, model=DummyModel())

    # Ensure model is loaded on CPU
    assert next(model_loaded.parameters()).device.type == "cpu"

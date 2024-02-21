# load_model_torch

# load_model_torch: Utility Function Documentation

## Introduction:

`load_model_torch` is a utility function in the `swarms.utils` library that is designed to load a saved PyTorch model and move it to the designated device. It provides flexibility allowing the user to specify the model file location, the device where the loaded model should be moved to, whether to strictly enforce the keys in the state dictionary to match the keys returned by the model's `state_dict()`, and many more.

Moreover, if the saved model file only contains the state dictionary, but not the model architecture, you can pass the model architecture as an argument. 

## Function Definition and Parameters:

```python
def load_model_torch(
    model_path: str = None,
    device: torch.device = None,
    model: nn.Module = None,
    strict: bool = True,
    map_location=None,
    *args,
    **kwargs,
) -> nn.Module:
```

The following table describes the parameters in detail:

|  Name  |  Type  | Default Value | Description |
| ------ | ------ | ------------- | ------------|
| model_path | str | None | A string specifying the path to the saved model file on disk. _Required_ |
| device | torch.device | None | A `torch.device` object that specifies the target device for the loaded model. If not provided, the function checks for the availability of a GPU and uses it if available. If not, it defaults to CPU. |
| model | nn.Module | None | An instance of `torch.nn.Module` representing the model's architecture. This parameter is required if the model file only contains the model's state dictionary and not the model architecture. |
| strict | bool | True | A boolean that determines whether to strictly enforce that the keys in the state dictionary match the keys returned by the model's `state_dict()` function. If set to `True`, the function will raise a KeyError when the state dictionary and `state_dict()` keys do not match. |
| map_location | callable | None | A function to remap the storage locations of the loaded model's parameters. Useful for loading models saved on a device type that is different from the current one. |
| *args, **kwargs | - | - | Additional arguments and keyword arguments to be passed to `torch.load`.

Returns: 

- `torch.nn.Module` - The loaded model after moving it to the desired device.

Raises:

- `FileNotFoundError` - If the saved model file is not found at the specified path.
- `RuntimeError` - If there was an error while loading the model.

## Example of Usage:

This function can be used directly inside your code as shown in the following examples:

### Example 1:
Loading a model without specifying a device results in the function choosing the most optimal available device automatically.

```python
import torch.nn as nn

from swarms.utils import load_model_torch

# Assume `mymodel.pth` is in the current directory
model_path = "./mymodel.pth"


# Define your model architecture if the model file only contains state dict
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


model = MyModel()

# Load the model
loaded_model = load_model_torch(model_path, model=model)

# Now you can use the loaded model for prediction or further training
```
### Example 2:
Explicitly specifying a device.

```python
# Assume `mymodel.pth` is in the current directory
model_path = "./mymodel.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
loaded_model = load_model_torch(model_path, device=device)
```

### Example 3:
Using a model file that contains only the state dictionary, not the model architecture.

```python
# Assume `mymodel_state_dict.pth` is in the current directory
model_path = "./mymodel_state_dict.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your model architecture
model = MyModel()

# Load the model
loaded_model = load_model_torch(model_path, device=device, model=model)
```

This gives you an insight on how to use `load_model_torch` utility function from `swarms.utils` library efficiently. Always remember to pass the model path argument while the other arguments can be optional based on your requirements. Furthermore, handle exceptions properly for smooth functioning of your PyTorch related projects.

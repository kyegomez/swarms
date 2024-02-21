# check_device

# Module/Function Name: check_device

The `check_device` is a utility function in PyTorch designed to identify and return the appropriate device(s) for CUDA processing. If CUDA is not available, a CPU device is returned. If CUDA is available, the function returns a list of all available GPU devices.

The function examines the CUDA availability, checks for multiple GPUs, and finds additional properties for each device.

## Function Signature and Arguments

**Signature:** 
```python
def check_device(
    log_level: Any = logging.INFO,
    memory_threshold: float = 0.8,
    capability_threshold: float = 3.5,
    return_type: str = "list",
) -> Union[torch.device, List[torch.device]]
```

| Parameter | Data Type | Default Value | Description |
| ------------- | ------------- | ------------- | ------------- |
| `log_level` | Any | logging.INFO | The log level. |
| `memory_threshold` | float | 0.8 | It is used to check the threshold of memory used on the GPU(s). |
| `capability_threshold` | float | 3.5 | It is used to consider only those GPU(s) which have higher compute capability compared to the threshold. |
| `return_type` | str | "list" | Depending on the `return_type` either a list of devices can be returned or a single device. |

This function does not take any mandatory argument. However, it supports optional arguments such as `log_level`, `memory_threshold`, `capability_threshold`, and `return_type`.

**Returns:**

- A single torch.device if one device or list of torch.devices if multiple CUDA devices are available, else returns the CPU device if CUDA is not available.

## Usage and Examples

### Example 1: Basic Usage 

```python
import logging

import torch

from swarms.utils import check_device

# Basic usage
device = check_device(
    log_level=logging.INFO,
    memory_threshold=0.8,
    capability_threshold=3.5,
    return_type="list",
)
```

### Example 2: Using CPU when CUDA is not available

```python
import torch

from swarms.utils import check_device

# When CUDA is not available
device = check_device()
print(device)  # If CUDA is not available it should return torch.device('cpu')
```

### Example 3: Multiple GPU Available

```python
import torch

from swarms.utils import check_device

# When multiple GPUs are available
device = check_device()
print(device)  # Should return a list of available GPU devices
```

## Tips and Additional Information

- This function is useful when a user wants to exploit CUDA capabilities for faster computation but unsure of the available devices. This function abstracts all the necessary checks and provides a list of CUDA devices to the user.
- The `memory_threshold` and `capability_threshold` are utilized to filter the GPU devices. The GPUs which have memory usage above the `memory_threshold` and compute capability below the `capability_threshold` are not considered.
- As of now, CPU does not have memory or capability values, therefore, in the respective cases, it will be returned as default without any comparison.

## Relevant Resources

- For more details about the CUDA properties functions used (`torch.cuda.get_device_capability, torch.cuda.get_device_properties`), please refer to the official PyTorch [CUDA semantics documentation](https://pytorch.org/docs/stable/notes/cuda.html).
- For more information about Torch device objects, you can refer to the official PyTorch [device documentation](https://pytorch.org/docs/stable/tensor_attributes.html#torch-device).
- For a better understanding of how the `logging` module works in Python, see the official Python [logging documentation](https://docs.python.org/3/library/logging.html).

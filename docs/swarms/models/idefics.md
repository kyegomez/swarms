# `Idefics` Documentation

## Introduction

Welcome to the documentation for Idefics, a versatile multimodal inference tool using pre-trained models from the Hugging Face Hub. Idefics is designed to facilitate the generation of text from various prompts, including text and images. This documentation provides a comprehensive understanding of Idefics, its architecture, usage, and how it can be integrated into your projects.

## Overview

Idefics leverages the power of pre-trained models to generate textual responses based on a wide range of prompts. It is capable of handling both text and images, making it suitable for various multimodal tasks, including text generation from images.

## Class Definition

```python
class Idefics:
    def __init__(
        self,
        checkpoint="HuggingFaceM4/idefics-9b-instruct",
        device=None,
        torch_dtype=torch.bfloat16,
        max_length=100,
    ):
```

## Usage

To use Idefics, follow these steps:

1. Initialize the Idefics instance:

```python
from swarms.models import Idefics

model = Idefics()
```

2. Generate text based on prompts:

```python
prompts = [
    "User: What is in this image? https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG"
]
response = model(prompts)
print(response)
```

### Example 1 - Image Questioning

```python
from swarms.models import Idefics

model = Idefics()
prompts = [
    "User: What is in this image? https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG"
]
response = model(prompts)
print(response)
```

### Example 2 - Bidirectional Conversation

```python
from swarms.models import Idefics

model = Idefics()
user_input = "User: What is in this image? https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG"
response = model.chat(user_input)
print(response)

user_input = "User: Who is that? https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052"
response = model.chat(user_input)
print(response)
```

### Example 3 - Configuration Changes

```python
model.set_checkpoint("new_checkpoint")
model.set_device("cpu")
model.set_max_length(200)
model.clear_chat_history()
```

## How Idefics Works

Idefics operates by leveraging pre-trained models from the Hugging Face Hub. Here's how it works:

1. **Initialization**: When you create an Idefics instance, it initializes the model using a specified checkpoint, sets the device for inference, and configures other parameters like data type and maximum text length.

2. **Prompt-Based Inference**: You can use the `infer` method to generate text based on prompts. It processes prompts in batched or non-batched mode, depending on your preference. It uses a pre-trained processor to handle text and images.

3. **Bidirectional Conversation**: The `chat` method enables bidirectional conversations. You provide user input, and the model responds accordingly. The chat history is maintained for context.

4. **Configuration Changes**: You can change the model checkpoint, device, maximum text length, or clear the chat history as needed during runtime.

## Parameters

- `checkpoint`: The name of the pre-trained model checkpoint (default is "HuggingFaceM4/idefics-9b-instruct").
- `device`: The device to use for inference. By default, it uses CUDA if available; otherwise, it uses CPU.
- `torch_dtype`: The data type to use for inference. By default, it uses torch.bfloat16.
- `max_length`: The maximum length of the generated text (default is 100).

## Additional Information

- Idefics provides a convenient way to engage in bidirectional conversations with pre-trained models.
- You can easily change the model checkpoint, device, and other settings to adapt to your specific use case.

That concludes the documentation for Idefics. We hope you find this tool valuable for your multimodal text generation tasks. If you have any questions or encounter any issues, please refer to the Hugging Face Transformers documentation for further assistance. Enjoy working with Idefics!
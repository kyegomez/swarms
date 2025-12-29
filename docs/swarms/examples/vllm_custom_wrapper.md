# Using vLLM with Custom Wrapper

This guide demonstrates how to create a custom wrapper class for vLLM (a high-performance LLM inference engine) and integrate it with Swarms agents. This approach gives you full control over the LLM interface and is ideal for local model deployments.

## What is vLLM?

vLLM is a fast and easy-to-use library for LLM inference and serving. It provides:

- High throughput serving with PagedAttention
- Efficient memory usage with continuous batching
- OpenAI-compatible API server
- Support for various model architectures

## Prerequisites

Install vLLM:

```bash
pip install vllm
```

## Creating a Custom vLLM Wrapper

The Agent class accepts an `llm` parameter that expects an object with a `run` method that takes a `task` parameter. Here's how to create a custom wrapper:

### Basic vLLM Wrapper

```python
from vllm import LLM, SamplingParams
from typing import Optional, Any


class VLLMWrapper:
    """
    Custom wrapper for vLLM to use with Swarms Agent.
    
    This wrapper implements the required interface for Swarms agents:
    - A `run` method that accepts a `task` parameter
    - Optional support for additional parameters like `img`, `stream`, etc.
    """
    
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ):
        """
        Initialize the vLLM wrapper.
        
        Args:
            model_name: Name or path of the model to load
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        # Initialize vLLM engine
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        
        # Create default sampling params
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    
    def run(
        self,
        task: str,
        img: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Run inference on the given task.
        
        This method implements the required interface for Swarms agents.
        
        Args:
            task: The input prompt/task to process
            img: Optional image input (not used in basic implementation)
            temperature: Override default temperature
            top_p: Override default top_p
            max_tokens: Override default max_tokens
            **kwargs: Additional parameters (ignored for basic implementation)
            
        Returns:
            str: The generated text response
        """
        # Create sampling params (use overrides if provided)
        sampling_params = SamplingParams(
            temperature=temperature or self.temperature,
            top_p=top_p or self.top_p,
            max_tokens=max_tokens or self.max_tokens,
        )
        
        # Generate response
        outputs = self.llm.generate([task], sampling_params)
        
        # Extract and return the generated text
        generated_text = outputs[0].outputs[0].text
        return generated_text
    
    def __call__(self, task: str, **kwargs) -> str:
        """
        Make the wrapper callable for convenience.
        
        Args:
            task: The input prompt/task to process
            **kwargs: Additional parameters
            
        Returns:
            str: The generated text response
        """
        return self.run(task, **kwargs)
```

## Using the vLLM Wrapper with Agent

### Basic Example

```python
from swarms import Agent
from vllm_wrapper import VLLMWrapper  # Your custom wrapper

# Initialize the vLLM wrapper
vllm_llm = VLLMWrapper(
    model_name="meta-llama/Llama-2-7b-chat-hf",  # Real Hugging Face model
    temperature=0.7,
    max_tokens=2048,
)

# Create agent with the custom LLM
agent = Agent(
    agent_name="vLLM-Agent",
    agent_description="Agent using vLLM for high-performance inference",
    llm=vllm_llm,  # Pass the custom wrapper
    system_prompt="You are a helpful AI assistant.",
    max_loops=1,
)

# Run the agent
response = agent.run("Explain quantum computing in simple terms.")
print(response)
```

### Advanced Example with Custom Configuration

```python
from swarms import Agent
from vllm_wrapper import VLLMWrapper

# Initialize vLLM with advanced settings
vllm_llm = VLLMWrapper(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",  # Real Mistral model
    tensor_parallel_size=2,  # Use 2 GPUs
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    temperature=0.8,
    top_p=0.95,
    max_tokens=4096,
)

# Create agent with streaming enabled
agent = Agent(
    agent_name="Advanced-vLLM-Agent",
    agent_description="High-performance agent with vLLM",
    llm=vllm_llm,
    system_prompt="You are an expert researcher and analyst.",
    max_loops=3,
    streaming_on=True,
    verbose=True,
)

# Run with a complex task
task = """
Analyze the following research question and provide a comprehensive answer:
What are the key differences between transformer architectures and 
recurrent neural networks in natural language processing?
"""

response = agent.run(task)
print(response)
```

## Complete Working Example

Here's a complete example that you can run:

```python
"""
Complete example: Using vLLM with Swarms Agent
"""

from vllm import LLM, SamplingParams
from typing import Optional
from swarms import Agent


class VLLMWrapper:
    """Custom vLLM wrapper for Swarms Agent."""
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ):
        """Initialize vLLM wrapper."""
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        # Initialize vLLM
        print(f"Loading vLLM model: {model_name}")
        self.llm = LLM(model=model_name)
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    
    def run(self, task: str, **kwargs) -> str:
        """Run inference on task."""
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        
        outputs = self.llm.generate([task], sampling_params)
        return outputs[0].outputs[0].text


def main():
    """Main function to run the example."""
    # Initialize vLLM wrapper
    vllm_llm = VLLMWrapper(
        model_name="meta-llama/Llama-2-7b-chat-hf",  # Real Llama 2 7B model
        temperature=0.7,
        max_tokens=1024,
    )
    
    # Create agent
    agent = Agent(
        agent_name="Research-Agent",
        agent_description="Agent for research and analysis tasks",
        llm=vllm_llm,
        system_prompt="You are a helpful research assistant.",
        max_loops=1,
        verbose=True,
    )
    
    # Run agent
    task = "What are the main advantages of using vLLM for LLM inference?"
    print("Running agent...")
    response = agent.run(task)
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
```

## Using vLLM with OpenAI-Compatible API Server

If you're running vLLM as a server, you can create a wrapper that connects to it:

```python
import requests
from typing import Optional


class VLLMServerWrapper:
    """
    Wrapper for vLLM server (OpenAI-compatible API).
    
    Use this when vLLM is running as a server:
    python -m vllm.entrypoints.openai.api_server --model <model_name>
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        Initialize vLLM server wrapper.
        
        Args:
            base_url: Base URL of the vLLM server
            model_name: Name of the model to use
            api_key: API key (usually not needed for local server)
            temperature: Default temperature
            max_tokens: Default max tokens
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def run(
        self,
        task: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Run inference via vLLM server.
        
        Args:
            task: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional parameters
            
        Returns:
            str: Generated text
        """
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": task}
            ],
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]


# Usage with server
vllm_server = VLLMServerWrapper(
    base_url="http://localhost:8000/v1",
    model_name="meta-llama/Llama-2-7b-chat-hf",  # Real Llama 2 7B model
)

agent = Agent(
    agent_name="Server-vLLM-Agent",
    llm=vllm_server,
    max_loops=1,
)

response = agent.run("Hello, how are you?")
print(response)
```

## Supported Models

vLLM supports many models from Hugging Face. Here are some real models you can use:

### Llama Models

```python
# Llama 2 7B (recommended for most use cases)
vllm_llm = VLLMWrapper(model_name="meta-llama/Llama-2-7b-chat-hf")

# Llama 2 13B (better quality, more memory)
vllm_llm = VLLMWrapper(model_name="meta-llama/Llama-2-13b-chat-hf")

# Llama 2 70B (best quality, requires multiple GPUs)
vllm_llm = VLLMWrapper(
    model_name="meta-llama/Llama-2-70b-chat-hf",
    tensor_parallel_size=4,  # Requires 4+ GPUs
)
```

### Mistral Models

```python
# Mistral 7B Instruct (fast and efficient)
vllm_llm = VLLMWrapper(model_name="mistralai/Mistral-7B-Instruct-v0.1")

# Mistral 7B v0.2 (updated version)
vllm_llm = VLLMWrapper(model_name="mistralai/Mistral-7B-Instruct-v0.2")
```

### Code Models

```python
# CodeLlama 7B (for code generation)
vllm_llm = VLLMWrapper(model_name="codellama/CodeLlama-7b-Instruct-hf")

# CodeLlama 13B (better code quality)
vllm_llm = VLLMWrapper(model_name="codellama/CodeLlama-13b-Instruct-hf")
```

### Small Models (Efficient)

```python
# Phi-2 (2.7B, very fast, good for simple tasks)
vllm_llm = VLLMWrapper(model_name="microsoft/phi-2")

# Phi-1.5 (1.3B, fastest, basic tasks)
vllm_llm = VLLMWrapper(model_name="microsoft/phi-1_5")
```

### Qwen Models

```python
# Qwen 7B Chat (good multilingual support)
vllm_llm = VLLMWrapper(model_name="Qwen/Qwen-7B-Chat")

# Qwen 14B Chat (better quality)
vllm_llm = VLLMWrapper(model_name="Qwen/Qwen-14B-Chat")
```

### Gemma Models

```python
# Gemma 7B IT (Google's model)
vllm_llm = VLLMWrapper(model_name="google/gemma-7b-it")

# Gemma 2B IT (smaller, faster)
vllm_llm = VLLMWrapper(model_name="google/gemma-2b-it")
```

## Best Practices

### 1. Memory Management

```python
# For large models, adjust GPU memory utilization
vllm_llm = VLLMWrapper(
    model_name="meta-llama/Llama-2-70b-chat-hf",  # Real Llama 2 70B model
    gpu_memory_utilization=0.8,  # Use 80% of GPU memory
)
```

### 2. Multi-GPU Setup

```python
# Use multiple GPUs for large models
vllm_llm = VLLMWrapper(
    model_name="meta-llama/Llama-2-70b-chat-hf",  # Real Llama 2 70B model
    tensor_parallel_size=4,  # Use 4 GPUs
)
```

### 3. Batch Processing

You can extend the wrapper to support batch processing:

```python
def run_batch(self, tasks: list[str], **kwargs) -> list[str]:
    """Process multiple tasks in batch."""
    sampling_params = SamplingParams(
        temperature=kwargs.get("temperature", self.temperature),
        top_p=kwargs.get("top_p", self.top_p),
        max_tokens=kwargs.get("max_tokens", self.max_tokens),
    )
    
    outputs = self.llm.generate(tasks, sampling_params)
    return [output.outputs[0].text for output in outputs]
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `gpu_memory_utilization` or use a smaller model
2. **Model Not Found**: Ensure the model path is correct or the model is available on Hugging Face
3. **Slow Inference**: Enable tensor parallelism for multi-GPU setups
4. **Import Errors**: Ensure vLLM is properly installed: `pip install vllm`

### Debug Tips

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test the wrapper directly
vllm_llm = VLLMWrapper(model_name="microsoft/phi-2")  # Real Phi-2 model
response = vllm_llm.run("Test prompt")
print(response)
```

## Conclusion

Using vLLM with Swarms agents via a custom wrapper provides:

- **High Performance**: Leverage vLLM's optimized inference engine
- **Full Control**: Customize the LLM interface to your needs
- **Local Deployment**: Run models on your own infrastructure
- **Flexibility**: Easy to extend with additional features

This approach is ideal for production deployments where you need high throughput and low latency inference.

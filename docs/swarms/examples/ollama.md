# Using Ollama with Swarms Agents

This guide demonstrates how to use Ollama (a tool for running large language models locally) with Swarms agents. You can use Ollama in two ways: the simple `model_name` approach or a custom wrapper for advanced control.

## What is Ollama?

Ollama is a tool for running large language models locally. It provides:

- Easy installation and setup
- Support for many open-source models
- Simple API interface
- No API keys required
- Runs entirely on your machine

## Prerequisites

Install Ollama and the Python client:

```bash
# Install Ollama (follow instructions at https://ollama.ai)
# Then install the Python client
pip install ollama
```

## Quick Start: Simple Approach

The easiest way to use Ollama is with the `model_name` parameter. This uses LiteLLM under the hood and follows [LiteLLM conventions](https://docs.litellm.ai/docs/providers/ollama).

```python
from swarms import Agent

# Initialize the agent with Ollama
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    model_name="ollama/llama2",  # Use ollama/ prefix
    system_prompt="Agent system prompt here",
    agent_description="Agent performs financial analysis.",
)

# Run a query
agent.run("What are the components of a startup's stock incentive equity plan?")
```

### Available Models

You can use any Ollama model with the `ollama/` prefix. Here are real models you can use:

```python
# Llama 2 models
agent = Agent(
    agent_name="Llama2-7B-Agent",
    model_name="ollama/llama2",  # 7B version (default)
    max_loops=1,
)

agent = Agent(
    agent_name="Llama2-13B-Agent",
    model_name="ollama/llama2:13b",  # 13B version
    max_loops=1,
)

# Mistral models
agent = Agent(
    agent_name="Mistral-Agent",
    model_name="ollama/mistral",  # 7B version
    max_loops=1,
)

# CodeLlama models
agent = Agent(
    agent_name="CodeLlama-Agent",
    model_name="ollama/codellama",  # 7B version
    max_loops=1,
)

agent = Agent(
    agent_name="CodeLlama-13B-Agent",
    model_name="ollama/codellama:13b",  # 13B version
    max_loops=1,
)

# Mixtral (mixture of experts)
agent = Agent(
    agent_name="Mixtral-Agent",
    model_name="ollama/mixtral:8x7b",  # 8x7B MoE model
    max_loops=1,
)

# Qwen models
agent = Agent(
    agent_name="Qwen-Agent",
    model_name="ollama/qwen:7b",  # 7B version
    max_loops=1,
)

# Neural Chat
agent = Agent(
    agent_name="NeuralChat-Agent",
    model_name="ollama/neural-chat:7b",  # 7B version
    max_loops=1,
)
```

## Advanced: Custom Wrapper Approach

For advanced use cases where you need full control over the Ollama interface, you can create a custom wrapper class. This approach gives you more flexibility and customization options.

### Creating a Custom Ollama Wrapper

The Agent class accepts an `llm` parameter that expects an object with a `run` method that takes a `task` parameter. Here's how to create a custom wrapper:

### Basic Ollama Wrapper

```python
import ollama
from typing import Optional, Any, Dict


class OllamaWrapper:
    """
    Custom wrapper for Ollama to use with Swarms Agent.
    
    This wrapper implements the required interface for Swarms agents:
    - A `run` method that accepts a `task` parameter
    - Optional support for additional parameters like `img`, `stream`, etc.
    """
    
    def __init__(
        self,
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        num_predict: int = 2048,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the Ollama wrapper.
        
        Args:
            model_name: Name of the Ollama model to use (e.g., "llama2", "mistral", "codellama")
            base_url: Base URL of the Ollama server
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            num_predict: Maximum number of tokens to generate
            system_prompt: Optional system prompt to use for all requests
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_predict = num_predict
        self.system_prompt = system_prompt
        
        # Initialize Ollama client
        self.client = ollama.Client(host=base_url)
        
        # Verify model is available
        self._verify_model()
    
    def _verify_model(self):
        """Verify that the model is available."""
        try:
            models = self.client.list()
            model_names = [model["name"] for model in models["models"]]
            
            if self.model_name not in model_names:
                print(f"Warning: Model '{self.model_name}' not found.")
                print(f"Available models: {', '.join(model_names)}")
                print(f"To pull a model, run: ollama pull {self.model_name}")
        except Exception as e:
            print(f"Warning: Could not verify model availability: {e}")
    
    def run(
        self,
        task: str,
        img: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        num_predict: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Run inference on the given task.
        
        This method implements the required interface for Swarms agents.
        
        Args:
            task: The input prompt/task to process
            img: Optional image input (for vision models)
            temperature: Override default temperature
            top_p: Override default top_p
            top_k: Override default top_k
            num_predict: Override default num_predict
            system_prompt: Override default system prompt
            **kwargs: Additional parameters
            
        Returns:
            str: The generated text response
        """
        # Prepare options
        options = {
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": top_p if top_p is not None else self.top_p,
            "top_k": top_k if top_k is not None else self.top_k,
            "num_predict": num_predict if num_predict is not None else self.num_predict,
        }
        
        # Prepare messages
        messages = []
        
        # Add system prompt if provided
        system = system_prompt or self.system_prompt
        if system:
            messages.append({
                "role": "system",
                "content": system
            })
        
        # Add user message
        user_message = {"role": "user", "content": task}
        
        # Add image if provided (for vision models)
        if img:
            user_message["images"] = [img]
        
        messages.append(user_message)
        
        # Generate response
        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            options=options,
        )
        
        # Extract and return the generated text
        return response["message"]["content"]
    
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

## When to Use Each Approach

| Approach              | Use Case                          | Pros                              | Cons                           |
|-----------------------|-----------------------------------|-----------------------------------|--------------------------------|
| **Simple (`model_name`)** | Quick setup, standard use cases | Easy to use, no extra code       | Less control, limited customization |
| **Custom Wrapper**    | Advanced features, custom logic   | Full control, extensible         | Requires more code             |

## Using the Custom Ollama Wrapper with Agent

### Basic Example

```python
from swarms import Agent
from ollama_wrapper import OllamaWrapper  # Your custom wrapper

# Initialize the Ollama wrapper
ollama_llm = OllamaWrapper(
    model_name="llama2",  # Or "mistral", "codellama", etc.
    temperature=0.7,
    num_predict=2048,
)

# Create agent with the custom LLM
agent = Agent(
    agent_name="Ollama-Agent",
    agent_description="Agent using Ollama for local inference",
    llm=ollama_llm,  # Pass the custom wrapper
    system_prompt="You are a helpful AI assistant.",
    max_loops=1,
)

# Run the agent
response = agent.run("Explain machine learning in simple terms.")
print(response)
```

### Advanced Example with Custom Configuration

```python
from swarms import Agent
from ollama_wrapper import OllamaWrapper

# Initialize Ollama with advanced settings
ollama_llm = OllamaWrapper(
    model_name="mistral",  # Use Mistral model
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    num_predict=4096,
    system_prompt="You are an expert researcher and analyst.",
)

# Create agent with streaming enabled
agent = Agent(
    agent_name="Advanced-Ollama-Agent",
    agent_description="High-performance agent with Ollama",
    llm=ollama_llm,
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
Complete example: Using Ollama with Swarms Agent
"""

import ollama
from typing import Optional
from swarms import Agent


class OllamaWrapper:
    """Custom Ollama wrapper for Swarms Agent."""
    
    def __init__(
        self,
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_predict: int = 2048,
        system_prompt: Optional[str] = None,
    ):
        """Initialize Ollama wrapper."""
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.num_predict = num_predict
        self.system_prompt = system_prompt
        
        # Initialize Ollama client
        print(f"Connecting to Ollama at {base_url}")
        self.client = ollama.Client(host=base_url)
        
        # Verify model
        self._verify_model()
    
    def _verify_model(self):
        """Verify that the model is available."""
        try:
            models = self.client.list()
            model_names = [model["name"] for model in models["models"]]
            
            if self.model_name not in model_names:
                print(f"Warning: Model '{self.model_name}' not found.")
                print(f"Available models: {', '.join(model_names)}")
                print(f"To pull a model, run: ollama pull {self.model_name}")
        except Exception as e:
            print(f"Warning: Could not verify model availability: {e}")
    
    def run(self, task: str, **kwargs) -> str:
        """Run inference on task."""
        options = {
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "num_predict": kwargs.get("num_predict", self.num_predict),
        }
        
        messages = []
        
        # Add system prompt if provided
        system = kwargs.get("system_prompt", self.system_prompt)
        if system:
            messages.append({
                "role": "system",
                "content": system
            })
        
        # Add user message
        messages.append({
            "role": "user",
            "content": task
        })
        
        # Generate response
        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            options=options,
        )
        
        return response["message"]["content"]


def main():
    """Main function to run the example."""
    # Initialize Ollama wrapper
    ollama_llm = OllamaWrapper(
        model_name="llama2",  # Change to your preferred model
        temperature=0.7,
        num_predict=1024,
    )
    
    # Create agent
    agent = Agent(
        agent_name="Research-Agent",
        agent_description="Agent for research and analysis tasks",
        llm=ollama_llm,
        system_prompt="You are a helpful research assistant.",
        max_loops=1,
        verbose=True,
    )
    
    # Run agent
    task = "What are the main advantages of using Ollama for local LLM inference?"
    print("Running agent...")
    response = agent.run(task)
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
```

## Using Different Ollama Models

Ollama supports many models. Here are examples with different models:

### Mistral Model

```python
ollama_llm = OllamaWrapper(
    model_name="mistral",  # Real Mistral 7B model
    temperature=0.8,
)

agent = Agent(
    agent_name="Mistral-Agent",
    llm=ollama_llm,
    max_loops=1,
)
```

### CodeLlama Model

```python
ollama_llm = OllamaWrapper(
    model_name="codellama",  # Real CodeLlama 7B model
    temperature=0.2,  # Lower temperature for code generation
)

agent = Agent(
    agent_name="Code-Agent",
    llm=ollama_llm,
    system_prompt="You are an expert programmer.",
    max_loops=1,
)

# Or use the 13B version for better code quality
ollama_llm = OllamaWrapper(
    model_name="codellama:13b",  # Real CodeLlama 13B model
    temperature=0.2,
)
```

### Llama 2 Model Variants

```python
# Llama 2 7B (default, fastest)
ollama_llm = OllamaWrapper(
    model_name="llama2",  # Real Llama 2 7B model
    temperature=0.7,
)

# Llama 2 13B (better quality)
ollama_llm = OllamaWrapper(
    model_name="llama2:13b",  # Real Llama 2 13B model
    temperature=0.7,
)

# Llama 2 70B (best quality, requires more memory)
ollama_llm = OllamaWrapper(
    model_name="llama2:70b",  # Real Llama 2 70B model
    temperature=0.7,
)
```

### Mixtral Model (Mixture of Experts)

```python
ollama_llm = OllamaWrapper(
    model_name="mixtral:8x7b",  # Real Mixtral 8x7B MoE model
    temperature=0.7,
)

agent = Agent(
    agent_name="Mixtral-Agent",
    llm=ollama_llm,
    max_loops=1,
)
```

### Qwen Model

```python
ollama_llm = OllamaWrapper(
    model_name="qwen:7b",  # Real Qwen 7B model
    temperature=0.7,
)

agent = Agent(
    agent_name="Qwen-Agent",
    llm=ollama_llm,
    max_loops=1,
)
```

## Streaming Support

You can extend the wrapper to support streaming:

```python
def run_stream(
    self,
    task: str,
    callback: Optional[callable] = None,
    **kwargs
):
    """
    Run inference with streaming support.
    
    Args:
        task: Input prompt
        callback: Optional callback function for each chunk
        **kwargs: Additional parameters
        
    Yields:
        str: Streaming text chunks
    """
    options = {
        "temperature": kwargs.get("temperature", self.temperature),
        "top_p": kwargs.get("top_p", self.top_p),
        "num_predict": kwargs.get("num_predict", self.num_predict),
    }
    
    messages = [{"role": "user", "content": task}]
    
    stream = self.client.chat(
        model=self.model_name,
        messages=messages,
        options=options,
        stream=True,
    )
    
    for chunk in stream:
        content = chunk["message"]["content"]
        if callback:
            callback(content)
        yield content
```

## Vision Models

Ollama supports vision models. Here's how to use them:

```python
class OllamaVisionWrapper(OllamaWrapper):
    """Extended wrapper for vision models."""
    
    def run(
        self,
        task: str,
        img: Optional[str] = None,
        **kwargs
    ) -> str:
        """Run inference with optional image input."""
        options = {
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "num_predict": kwargs.get("num_predict", self.num_predict),
        }
        
        messages = [{"role": "user", "content": task}]
        
        # Add image if provided
        if img:
            # Read image file and encode as base64
            import base64
            with open(img, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            messages[0]["images"] = [image_data]
        
        response = self.client.chat(
            model=self.model_name,  # Use vision model like "llava"
            messages=messages,
            options=options,
        )
        
        return response["message"]["content"]


# Usage with vision
vision_llm = OllamaVisionWrapper(
    model_name="llava",  # Vision model
)

agent = Agent(
    agent_name="Vision-Agent",
    llm=vision_llm,
    max_loops=1,
)

response = agent.run(
    "Describe this image in detail.",
    img="path/to/image.jpg"
)
```

## Best Practices

### 1. Model Selection

Choose the right model for your task:

- **llama2** or **llama2:13b**: General purpose, good balance (7B or 13B)
- **mistral**: Fast and efficient (7B)
- **codellama** or **codellama:13b**: Best for code generation (7B or 13B)
- **mixtral:8x7b**: High quality, mixture of experts (8x7B)
- **llava** or **llava:13b**: For vision tasks (7B or 13B)
- **qwen:7b** or **qwen:14b**: Good multilingual support (7B or 14B)
- **phi** or **phi:2.7b**: Small and fast (2.7B)
- **gemma:2b** or **gemma:7b**: Google's efficient models (2B or 7B)

### 2. Temperature Settings

```python
# For creative tasks
ollama_llm = OllamaWrapper(temperature=0.9)

# For factual tasks
ollama_llm = OllamaWrapper(temperature=0.3)

# For code generation
ollama_llm = OllamaWrapper(temperature=0.2)
```

### 3. Error Handling

Add error handling to your wrapper:

```python
def run(self, task: str, **kwargs) -> str:
    """Run inference with error handling."""
    try:
        # ... existing code ...
        response = self.client.chat(...)
        return response["message"]["content"]
    except Exception as e:
        print(f"Error in Ollama inference: {e}")
        raise
```

## Troubleshooting

### Common Issues

1. **Model Not Found**: Pull the model first: `ollama pull llama2`
2. **Connection Error**: Ensure Ollama is running: `ollama serve`
3. **Out of Memory**: Use a smaller model or reduce `num_predict`
4. **Slow Inference**: Use a faster model like `mistral` or reduce model size

### Debug Tips

```python
# Test Ollama connection
import ollama
client = ollama.Client()
models = client.list()
print("Available models:", [m["name"] for m in models["models"]])

# Test the wrapper directly
ollama_llm = OllamaWrapper(model_name="llama2")  # Real Llama 2 7B model
response = ollama_llm.run("Test prompt")
print(response)

# Or test with a different model
ollama_llm = OllamaWrapper(model_name="mistral")  # Real Mistral 7B model
response = ollama_llm.run("Test prompt")
print(response)
```

## Real Ollama Models Reference

Common Ollama models you can use (all are real, available models):

| Model                | Description                    | Pull Command                    |
|----------------------|--------------------------------|---------------------------------|
| `llama2`             | Meta's Llama 2 (7B default)   | `ollama pull llama2`            |
| `llama2:13b`         | Llama 2 13B version            | `ollama pull llama2:13b`        |
| `llama2:70b`         | Llama 2 70B version            | `ollama pull llama2:70b`        |
| `mistral`            | Mistral AI 7B model            | `ollama pull mistral`           |
| `codellama`          | Code-focused Llama 7B          | `ollama pull codellama`         |
| `codellama:13b`      | CodeLlama 13B                  | `ollama pull codellama:13b`     |
| `codellama:34b`      | CodeLlama 34B                  | `ollama pull codellama:34b`     |
| `mixtral:8x7b`       | Mixtral 8x7B MoE model         | `ollama pull mixtral:8x7b`      |
| `llava`              | Vision-language model 7B       | `ollama pull llava`             |
| `llava:13b`          | LLaVA 13B version               | `ollama pull llava:13b`         |
| `phi`                | Microsoft's Phi-2 (2.7B)       | `ollama pull phi`               |
| `phi:2.7b`           | Phi-2 2.7B                     | `ollama pull phi:2.7b`          |
| `gemma:2b`           | Google's Gemma 2B               | `ollama pull gemma:2b`          |
| `gemma:7b`           | Google's Gemma 7B               | `ollama pull gemma:7b`          |
| `qwen:7b`            | Qwen 7B model                  | `ollama pull qwen:7b`           |
| `qwen:14b`           | Qwen 14B model                 | `ollama pull qwen:14b`          |
| `neural-chat:7b`     | Neural Chat 7B                 | `ollama pull neural-chat:7b`    |
| `starling-lm:7b`     | Starling LM 7B                 | `ollama pull starling-lm:7b`    |
| `orca-mini:3b`       | Orca Mini 3B                   | `ollama pull orca-mini:3b`      |

Pull a model with: `ollama pull <model_name>`

For example:

```bash
ollama pull llama2
ollama pull mistral
ollama pull codellama:13b
```

## Conclusion

Using Ollama with Swarms agents provides:

- **Local Deployment**: Run models entirely on your machine
- **No API Keys**: No need for external API keys
- **Privacy**: All data stays local
- **Full Control**: Customize the interface to your needs (with custom wrapper)
- **Easy Setup**: Simple installation and configuration

This approach is ideal for:

- Development and testing
- Privacy-sensitive applications
- Offline deployments
- Cost-effective local inference

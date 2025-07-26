# Environment Variables

## Overview

Swarms uses environment variables for configuration management and secure credential storage. This approach keeps sensitive information like API keys out of your code and allows for easy configuration changes across different environments.

## Core Environment Variables

### Framework Configuration

=== "Configuration Variables"

    | Variable | Description | Example |
    |----------|-------------|---------|
    | `SWARMS_VERBOSE_GLOBAL` | Controls global logging verbosity | `True` or `False` |
    | `WORKSPACE_DIR` | Defines the workspace directory for agent operations | `agent_workspace` |

### LLM Provider API Keys

=== "OpenAI"
    ```bash
    OPENAI_API_KEY="your-openai-key"
    ```

=== "Anthropic"
    ```bash
    ANTHROPIC_API_KEY="your-anthropic-key"
    ```

=== "Groq"
    ```bash
    GROQ_API_KEY="your-groq-key"
    ```

=== "Google"
    ```bash
    GEMINI_API_KEY="your-gemini-key"
    ```

=== "Hugging Face"
    ```bash
    HUGGINGFACE_TOKEN="your-huggingface-token"
    ```

=== "Perplexity AI"
    ```bash
    PPLX_API_KEY="your-perplexity-key"
    ```

=== "AI21"
    ```bash
    AI21_API_KEY="your-ai21-key"
    ```

=== "Cohere"
    ```bash
    COHERE_API_KEY="your-cohere-key"
    ```

=== "Mistral AI"
    ```bash
    MISTRAL_API_KEY="your-mistral-key"
    ```

=== "Together AI"
    ```bash
    TOGETHER_API_KEY="your-together-key"
    ```

### Tool Provider Keys

=== "Search Tools"
    ```bash
    BING_BROWSER_API="your-bing-key"
    BRAVESEARCH_API_KEY="your-brave-key"
    TAVILY_API_KEY="your-tavily-key"
    YOU_API_KEY="your-you-key"
    ```

=== "Analytics & Monitoring"
    ```bash
    EXA_API_KEY="your-exa-key"
    ```

=== "Browser Automation"
    ```bash
    MULTION_API_KEY="your-multion-key"
    ```

## Security Best Practices

### Environment File Management

1. Create a `.env` file in your project root
2. Never commit `.env` files to version control
3. Add `.env` to your `.gitignore`:
    ```bash
    echo ".env" >> .gitignore
    ```

### API Key Security

!!! warning "Important Security Considerations"
    - Rotate API keys regularly
    - Use different API keys for development and production
    - Never hardcode API keys in your code
    - Limit API key permissions to only what's necessary
    - Monitor API key usage for unusual patterns

### Template Configuration

Create a `.env.example` template without actual values:

```bash
# Required Configuration
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
GROQ_API_KEY=""
WORKSPACE_DIR="agent_workspace"

# Optional Configuration
SWARMS_VERBOSE_GLOBAL="False"
```

### Loading Environment Variables

```python
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Access variables
workspace_dir = os.getenv("WORKSPACE_DIR")
openai_key = os.getenv("OPENAI_API_KEY")
```

## Environment Setup Guide

=== "1. Install Dependencies"
    ```bash
    pip install python-dotenv
    ```

=== "2. Create Environment File"
    ```bash
    cp .env.example .env
    ```

=== "3. Configure Variables"
    - Open `.env` in your text editor
    - Add your API keys and configuration
    - Save the file

=== "4. Verify Setup"
    ```python
    import os
    from dotenv import load_dotenv

    load_dotenv()
    assert os.getenv("OPENAI_API_KEY") is not None, "OpenAI API key not found"
    ```

## Environment-Specific Configuration

=== "Development"
    ```bash
    WORKSPACE_DIR="agent_workspace"
    SWARMS_VERBOSE_GLOBAL="True"
    ```

=== "Production"
    ```bash
    WORKSPACE_DIR="/var/swarms/workspace"
    SWARMS_VERBOSE_GLOBAL="False"
    ```

=== "Testing"
    ```bash
    WORKSPACE_DIR="test_workspace"
    SWARMS_VERBOSE_GLOBAL="True"
    ```

## Troubleshooting

### Common Issues

???+ note "Environment Variables Not Loading"
    - Verify `.env` file exists in project root
    - Confirm `load_dotenv()` is called before accessing variables
    - Check file permissions

???+ note "API Key Issues"
    - Verify key format is correct
    - Ensure key has not expired
    - Check for leading/trailing whitespace

???+ note "Workspace Directory Problems"
    - Confirm directory exists
    - Verify write permissions
    - Check path is absolute when required

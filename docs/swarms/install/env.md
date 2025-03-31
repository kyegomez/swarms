# Environment Variable Management & Security

This guide provides comprehensive documentation for managing environment variables and API keys securely in the Swarms framework.

## Overview

Swarms uses environment variables for configuration management and secure credential storage. This approach keeps sensitive information like API keys out of your code and allows for easy configuration changes across different environments.

## Core Environment Variables

### Framework Configuration

- `SWARMS_VERBOSE_GLOBAL`: Controls global logging verbosity
  ```bash
  SWARMS_VERBOSE_GLOBAL="True"  # Enable verbose logging
  SWARMS_VERBOSE_GLOBAL="False" # Disable verbose logging
  ```

- `WORKSPACE_DIR`: Defines the workspace directory for agent operations
  ```bash
  WORKSPACE_DIR="agent_workspace"
  ```

### API Keys

#### Model Provider Keys

1. **OpenAI**
   - `OPENAI_API_KEY`: Authentication for GPT models
   ```bash
   OPENAI_API_KEY="your-openai-key"
   ```

2. **Anthropic**
   - `ANTHROPIC_API_KEY`: Authentication for Claude models
   ```bash
   ANTHROPIC_API_KEY="your-anthropic-key"
   ```

3. **Google**
   - `GEMINI_API_KEY`: Authentication for Gemini models

4. **Hugging Face**
   - `HUGGINGFACE_TOKEN`: Access to Hugging Face models

5. **Perplexity AI**
   - `PPLX_API_KEY`: Access to Perplexity models

6. **AI21**
   - `AI21_API_KEY`: Access to AI21 models

#### Tool Provider Keys

1. **Search Tools**
   - `BING_BROWSER_API`: Bing search capabilities
   - `BRAVESEARCH_API_KEY`: Brave search integration
   - `TAVILY_API_KEY`: Tavily search services
   - `YOU_API_KEY`: You.com search integration

2. **Analytics & Monitoring**
   - `EXA_API_KEY`: Exa.ai services

3. **Browser Automation**
   - `MULTION_API_KEY`: Multi-browser automation

   
## Security Best Practices

### 1. Environment File Management

- Create a `.env` file in your project root
- Never commit `.env` files to version control
- Add `.env` to your `.gitignore`:
  ```bash
  echo ".env" >> .gitignore
  ```

### 2. API Key Security

- Rotate API keys regularly
- Use different API keys for development and production
- Never hardcode API keys in your code
- Limit API key permissions to only what's necessary
- Monitor API key usage for unusual patterns

### 3. Template Configuration

Create a `.env.example` template without actual values:

```bash
# Required Configuration
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
WORKSPACE_DIR="agent_workspace"

# Optional Configuration
SWARMS_VERBOSE_GLOBAL="False"
```

### 4. Loading Environment Variables

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

1. **Install Dependencies**:
   ```bash
   pip install python-dotenv
   ```

2. **Create Environment File**:
   ```bash
   cp .env.example .env
   ```

3. **Configure Variables**:
   - Open `.env` in your text editor
   - Add your API keys and configuration
   - Save the file

4. **Verify Setup**:
   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()
   assert os.getenv("OPENAI_API_KEY") is not None, "OpenAI API key not found"
   ```

## Environment-Specific Configuration

### Development

```bash
WORKSPACE_DIR="agent_workspace"
SWARMS_VERBOSE_GLOBAL="True"
```

### Production

```bash
WORKSPACE_DIR="/var/swarms/workspace"
SWARMS_VERBOSE_GLOBAL="False"
```

### Testing

```bash
WORKSPACE_DIR="test_workspace"
SWARMS_VERBOSE_GLOBAL="True"
```

## Troubleshooting

### Common Issues

1. **Environment Variables Not Loading**
   - Verify `.env` file exists in project root
   - Confirm `load_dotenv()` is called before accessing variables
   - Check file permissions

2. **API Key Issues**
   - Verify key format is correct
   - Ensure key has not expired
   - Check for leading/trailing whitespace

3. **Workspace Directory Problems**
   - Confirm directory exists
   - Verify write permissions
   - Check path is absolute when required

## Additional Resources

- [Swarms Documentation](https://docs.swarms.world)
- [Security Best Practices](https://swarms.world/security)
- [API Documentation](https://swarms.world/docs/api)


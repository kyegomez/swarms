# Swarms API Playground Documentation

## Overview

The Swarms Playground (`https://swarms.world/platform/playground`) is an interactive testing environment that allows you to experiment with the Swarms API in real-time. This powerful tool enables you to configure AI agents, test different parameters, and generate code examples in multiple programming languages without writing any code manually.

## Key Features

- **Real-time API Testing**: Execute Swarms API calls directly in the browser

- **Multi-language Code Generation**: Generate code examples in Python, Rust, Go, and TypeScript

- **Interactive Configuration**: Visual interface for setting up agent parameters

- **Live Output**: See API responses immediately in the output terminal

- **Code Export**: Copy generated code for use in your applications


## Interface Overview

### Language Selection

The playground supports code generation in four programming languages:

- **Python**: Default language with `requests` library implementation

- **Rust**: Native Rust HTTP client implementation

- **Go**: Standard Go HTTP package implementation  

- **TypeScript**: Node.js/browser-compatible implementation


Switch between languages using the dropdown menu in the top-right corner to see language-specific code examples.

### Agent Modes

The playground offers two distinct modes for testing different types of AI implementations:

#### Single Agent Mode
Test individual AI agents with specific configurations and tasks. Ideal for:
- Prototype testing

- Parameter optimization

- Simple task automation

- API familiarization


#### Multi-Agent Mode  
Experiment with coordinated AI agent systems. Perfect for:
- Complex workflow automation

- Collaborative AI systems

- Distributed task processing

- Advanced orchestration scenarios


## Configuration Parameters

### Basic Agent Settings

#### Agent Name
**Purpose**: Unique identifier for your agent
**Usage**: Helps distinguish between different agent configurations
**Example**: `"customer_service_bot"`, `"data_analyst"`, `"content_writer"`

#### Model Name
**Purpose**: Specifies which AI model to use for the agent
**Default**: `gpt-4o-mini`
**Options**: Various OpenAI and other supported models
**Impact**: Affects response quality, speed, and cost

#### Description
**Purpose**: Human-readable description of the agent's purpose
**Usage**: Documentation and identification
**Best Practice**: Be specific about the agent's intended function

#### System Prompt
**Purpose**: Core instructions that define the agent's behavior and personality
**Impact**: Critical for agent performance and response style
**Tips**: 
- Be clear and specific

- Include role definition

- Specify output format if needed

- Add relevant constraints


### Advanced Parameters

#### Temperature
**Range**: 0.0 - 2.0

**Default**: 0.5
**Purpose**: Controls randomness in responses
- **Low (0.0-0.3)**: More deterministic, consistent responses

- **Medium (0.4-0.7)**: Balanced creativity and consistency  

- **High (0.8-2.0)**: More creative and varied responses


#### Max Tokens
**Default**: 8192
**Purpose**: Maximum length of the agent's response
**Considerations**:
- Higher values allow longer responses

- Impacts API costs

- Model-dependent limits apply


#### Role
**Default**: `worker`
**Purpose**: Defines the agent's role in multi-agent scenarios
**Common Roles**: `worker`, `manager`, `coordinator`, `specialist`

#### Max Loops
**Default**: 1
**Purpose**: Number of iterations the agent can perform
**Usage**:
- `1`: Single response

- `>1`: Allows iterative problem solving


#### MCP URL (Optional)
**Purpose**: Model Context Protocol URL for external integrations
**Usage**: Connect to external services or data sources
**Format**: Valid URL pointing to MCP-compatible service

### Task Definition

#### Task
**Purpose**: Specific instruction or query for the agent to process
**Best Practices**:
- Be specific and clear

- Include all necessary context

- Specify desired output format

- Provide examples when helpful


## Using the Playground

### Step-by-Step Guide

1. **Select Mode**: Choose between Single Agent or Multi-Agent
2. **Choose Language**: Select your preferred programming language
3. **Configure Agent**: Fill in the required parameters
4. **Define Task**: Enter your specific task or query
5. **Run Agent**: Click the "Run Agent" button
6. **Review Output**: Check the Output Terminal for results
7. **Copy Code**: Use the generated code in your applications

### Testing Strategies

#### Parameter Experimentation

- **Temperature Testing**: Try different temperature values to find optimal creativity levels

- **Prompt Engineering**: Iterate on system prompts to improve responses

- **Token Optimization**: Adjust max_tokens based on expected response length


#### Workflow Development

- **Start Simple**: Begin with basic tasks and gradually increase complexity

- **Iterative Refinement**: Use playground results to refine your approach

- **Documentation**: Keep notes on successful configurations


## Output Interpretation

### Output Terminal

The Output Terminal displays:

- **Agent Responses**: Direct output from the AI agent

- **Error Messages**: API errors or configuration issues

- **Execution Status**: Success/failure indicators

- **Response Metadata**: Token usage, timing information


### Code Preview

The Code Preview section shows:

- **Complete Implementation**: Ready-to-use code in your selected language

- **API Configuration**: Proper headers and authentication setup

- **Request Structure**: Correctly formatted payload

- **Response Handling**: Basic error handling and output processing


## Code Examples by Language

### Python Implementation
```python
import requests

url = "https://swarms-api-285321057562.us-east1.run.app/v1/agent/completions"
headers = {
    "Content-Type": "application/json",
    "x-api-key": "your-api-key-here"
}

payload = {
    "agent_config": {
        "agent_name": "example_agent",
        "description": "Example agent for demonstration",
        "system_prompt": "You are a helpful assistant.",
        "model_name": "gpt-4o-mini",
        "auto_generate_prompt": false,
        "max_tokens": 8192,
        "temperature": 0.5,
        "role": "worker",
        "max_loops": 1,
        "tools_list_dictionary": null,
        "mcp_url": null
    },
    "task": "Explain quantum computing in simple terms"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

### Key Code Components

#### API Endpoint

- **URL**: `https://swarms-api-285321057562.us-east1.run.app/v1/agent/completions`

- **Method**: POST

- **Authentication**: API key in `x-api-key` header


#### Request Structure

- **Headers**: Content-Type and API key

- **Payload**: Agent configuration and task

- **Response**: JSON with agent output and metadata


## Best Practices

### Security

- **API Key Management**: Never expose API keys in client-side code

- **Environment Variables**: Store sensitive credentials securely

- **Rate Limiting**: Respect API rate limits in production


### Performance Optimization

- **Parameter Tuning**: Optimize temperature and max_tokens for your use case

- **Prompt Engineering**: Craft efficient system prompts

- **Caching**: Implement response caching for repeated queries


### Development Workflow

- **Prototype in Playground**: Test configurations before implementation

- **Document Successful Configs**: Save working parameter combinations

- **Iterate and Improve**: Use playground for continuous optimization


## Troubleshooting

### Common Issues

#### No Output in Terminal

- **Check API Key**: Ensure valid API key is configured

- **Verify Parameters**: All required fields must be filled

- **Network Issues**: Check internet connection


#### Unexpected Responses

- **Review System Prompt**: Ensure clear instructions

- **Adjust Temperature**: Try different creativity levels

- **Check Task Definition**: Verify task clarity and specificity


#### Code Generation Issues

- **Language Selection**: Ensure correct language is selected

- **Copy Functionality**: Use the "Copy Code" button for accurate copying

- **Syntax Validation**: Test generated code in your development environment


## Integration Guide

### From Playground to Production

1. **Copy Generated Code**: Use the Code Preview section
2. **Add Error Handling**: Implement robust error handling
3. **Configure Environment**: Set up proper API key management
4. **Test Thoroughly**: Validate in your target environment
5. **Monitor Performance**: Track API usage and response quality

The Swarms Playground is your gateway to understanding and implementing the Swarms API effectively. Use it to experiment, learn, and build confidence before deploying AI agents in production environments.
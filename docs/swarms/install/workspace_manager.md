# Swarms Framework Environment Configuration

This guide details the environment variables used in the Swarms framework for configuration and customization of your agent-based applications.

## Configuration Setup

Create a `.env` file in your project's root directory to configure the Swarms framework. This file will contain all necessary environment variables for customizing your agent's behavior, logging, and analytics.

## Environment Variables

### Core Variables

#### `WORKSPACE_DIR`
- **Purpose**: Defines the directory where all agent states and execution logs are stored
- **Type**: String (path)
- **Default**: `./workspace`
- **Example**: 
```bash
WORKSPACE_DIR=/path/to/your/workspace
```
- **Usage**:
  - Stores JSON files containing agent states
  - Maintains execution history
  - Keeps track of agent interactions
  - Preserves conversation logs

### Telemetry Configuration

#### `USE_TELEMETRY`
- **Purpose**: Controls whether telemetry data is collected
- **Type**: Boolean
- **Default**: `false`
- **Example**:
```bash
USE_TELEMETRY=true
```
- **Data Collected**:
  - Agent performance metrics
  - Execution time statistics
  - Memory usage
  - Error rates
  - System health indicators

### Analytics Integration

#### `SWARMS_API_KEY`
- **Purpose**: Authentication key for the Swarms Analytics Suite
- **Type**: String
- **Required**: Yes, for analytics features
- **Example**:
```bash
SWARMS_API_KEY=your_api_key_here
```
- **Features**:
  - Real-time agent execution tracking
  - Usage analytics
  - Performance monitoring
  - Cost tracking
  - Custom metrics

## Getting Started

1. Create a new `.env` file:
```bash
touch .env
```

2. Add your configuration:
```bash
# Basic configuration
WORKSPACE_DIR=./my_workspace

# Enable telemetry
USE_TELEMETRY=true

# Add your Swarms API key
SWARMS_API_KEY=your_api_key_here
```

3. Obtain your API key:
   - Visit [swarms.ai dashboard](https://swarms.ai)
   - Create an account or log in
   - Navigate to the API section
   - Generate your unique API key

## Best Practices

1. **Security**:
   - Never commit your `.env` file to version control
   - Add `.env` to your `.gitignore` file
   - Keep your API keys secure and rotate them periodically

2. **Workspace Organization**:
   - Use descriptive workspace directory names
   - Implement regular cleanup of old logs
   - Monitor workspace size to prevent disk space issues

3. **Telemetry Management**:
   - Enable telemetry in development for debugging
   - Consider privacy implications in production
   - Review collected data periodically

## Examples

### Basic Development Setup
```bash
WORKSPACE_DIR=./dev_workspace
USE_TELEMETRY=true
SWARMS_API_KEY=sk_test_xxxxxxxxxxxx
```

### Production Setup
```bash
WORKSPACE_DIR=/var/log/swarms/prod_workspace
USE_TELEMETRY=true
SWARMS_API_KEY=sk_prod_xxxxxxxxxxxx
```

## Troubleshooting

Common issues and solutions:

1. **Workspace Access Issues**:
   - Ensure proper file permissions
   - Verify the directory exists
   - Check disk space availability

2. **API Key Problems**:
   - Confirm key is properly formatted
   - Verify key hasn't expired
   - Check for proper environment variable loading

3. **Telemetry Issues**:
   - Confirm network connectivity
   - Verify firewall settings
   - Check for proper boolean values

## Additional Resources

- [Swarms Framework Documentation](https://github.com/kyegomez/swarms)
- [Swarms Analytics Dashboard](https://swarms.ai)
- [API Reference](https://swarms.ai/docs/api)
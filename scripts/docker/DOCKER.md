# Swarms Docker Image

This repository includes a Docker image for running Swarms, an AI agent framework. The image is automatically built and published to DockerHub on every push to the main branch and on version tags.

## 🐳 Quick Start

### Pull and Run

```bash
# Pull the latest image
docker pull kyegomez/swarms:latest

# Run a simple test
docker run --rm kyegomez/swarms:latest python test_docker.py

# Run with interactive shell
docker run -it --rm kyegomez/swarms:latest bash
```

### Using Specific Versions

```bash
# Pull a specific version
docker pull kyegomez/swarms:v8.0.4

# Run with specific version
docker run --rm kyegomez/swarms:v8.0.4 python -c "import swarms; print(swarms.__version__)"
```

## 🏗️ Building Locally

### Prerequisites

- Docker installed on your system
- Git to clone the repository

### Build Steps

```bash
# Clone the repository
git clone https://github.com/kyegomez/swarms.git
cd swarms

# Build the image
docker build -t swarms:latest .

# Test the image
docker run --rm swarms:latest python test_docker.py
```

## 🚀 Usage Examples

### Basic Agent Example

```bash
# Create a Python script (agent_example.py)
cat > agent_example.py << 'EOF'
from swarms import Agent

# Create an agent
agent = Agent(
    agent_name="test_agent",
    system_prompt="You are a helpful AI assistant."
)

# Run the agent
result = agent.run("Hello! How are you today?")
print(result)
EOF

# Run in Docker
docker run --rm -v $(pwd):/app swarms:latest python /app/agent_example.py
```

### Interactive Development

```bash
# Run with volume mount for development
docker run -it --rm \
  -v $(pwd):/app \
  -w /app \
  swarms:latest bash

# Inside the container, you can now run Python scripts
python your_script.py
```

### Using Environment Variables

```bash
# Run with environment variables
docker run --rm \
  -e OPENAI_API_KEY=your_api_key_here \
  -e ANTHROPIC_API_KEY=your_anthropic_key_here \
  swarms:latest python your_script.py
```

## 🔧 Configuration

### Environment Variables

The Docker image supports the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `GOOGLE_API_KEY`: Your Google API key
- `PYTHONPATH`: Additional Python path entries
- `PYTHONUNBUFFERED`: Set to 1 for unbuffered output

### Volume Mounts

Common volume mount patterns:

```bash
# Mount current directory for development
-v $(pwd):/app

# Mount specific directories
-v $(pwd)/data:/app/data
-v $(pwd)/models:/app/models

# Mount configuration files
-v $(pwd)/config:/app/config
```

## 🐛 Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   # Fix permission issues
   docker run --rm -v $(pwd):/app:rw swarms:latest python your_script.py
   ```

2. **Memory Issues**
   ```bash
   # Increase memory limit
   docker run --rm --memory=4g swarms:latest python your_script.py
   ```

3. **Network Issues**
   ```bash
   # Use host network
   docker run --rm --network=host swarms:latest python your_script.py
   ```

### Debug Mode

```bash
# Run with debug output
docker run --rm -e PYTHONUNBUFFERED=1 swarms:latest python -u your_script.py

# Run with interactive debugging
docker run -it --rm swarms:latest python -m pdb your_script.py
```

## 🔄 CI/CD Integration

The Docker image is automatically built and published via GitHub Actions:

- **Triggers**: Push to main branch, version tags (v*.*.*)
- **Platforms**: linux/amd64, linux/arm64
- **Registry**: DockerHub (kyegomez/swarms)

### GitHub Actions Secrets Required

- `DOCKERHUB_USERNAME`: Your DockerHub username
- `DOCKERHUB_TOKEN`: Your DockerHub access token

## 📊 Image Details

### Base Image
- Python 3.11-slim-bullseye
- Multi-stage build for optimization
- UV package manager for faster installations

### Image Size
- Optimized for minimal size
- Multi-stage build reduces final image size
- Only necessary dependencies included

### Security
- Non-root user execution
- Minimal system dependencies
- Regular security updates

## 🤝 Contributing

To contribute to the Docker setup:

1. Fork the repository
2. Make your changes to the Dockerfile
3. Test locally: `docker build -t swarms:test .`
4. Submit a pull request

### Testing Changes

```bash
# Build test image
docker build -t swarms:test .

# Run tests
docker run --rm swarms:test python test_docker.py

# Test with your code
docker run --rm -v $(pwd):/app swarms:test python your_test_script.py
```

## 📝 License

This Docker setup is part of the Swarms project and follows the same MIT license.

## 🆘 Support

For issues with the Docker image:

1. Check the troubleshooting section above
2. Review the GitHub Actions logs for build issues
3. Open an issue on GitHub with detailed error information
4. Include your Docker version and system information

---

**Note**: This Docker image is automatically updated with each release. For production use, consider pinning to specific version tags for stability. 

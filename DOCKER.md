# Docker Setup for Swarms (x86_64/amd64)

This repository includes Docker support for running Swarms on x86_64 (amd64) architecture.

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and run the container
docker-compose up -d

# View logs
docker-compose logs -f swarms

# Stop the container
docker-compose down
```

### Option 2: Using Docker Build

```bash
# Build the image
docker build --platform linux/amd64 -t swarms:latest .

# Run the container
docker run --platform linux/amd64 -it swarms:latest bash

# Run with environment variables
docker run --platform linux/amd64 \
  -e OPENAI_API_KEY=your_key_here \
  -e ANTHROPIC_API_KEY=your_key_here \
  -v $(pwd):/app \
  swarms:latest
```

### Option 3: Pull from Docker Hub

```bash
# Pull the pre-built image
docker pull swarmscorp/swarms:latest

# Note: The pre-built image may need to be updated for amd64 compatibility
# Use the local build method above for guaranteed amd64 support
```

## Available Services

The `docker-compose.yml` includes two services:

### 1. swarms (Default Service)
- Runs a basic swarms container
- Includes health checks
- Auto-restarts unless stopped

### 2. swarms-dev (Development Service)
- Interactive bash shell
- Ideal for development and testing
- Mounted volumes for live code changes

```bash
# Run development container
docker-compose up -d swarms-dev

# Attach to the container
docker exec -it swarms-dev-container bash
```

## Architecture

This Docker image is explicitly built for **x86_64 (amd64)** architecture using:
- Base image: `python:3.11-slim-bullseye`
- Platform: `linux/amd64`
- Multi-stage build for optimized image size

## Environment Variables

Configure your Swarms instance by setting environment variables:

```bash
# Create a .env file in the root directory
cat > .env << EOF
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
EOF
```

Then use with docker-compose:

```bash
docker-compose --env-file .env up -d
```

## Volume Mounts

The docker-compose configuration mounts the following directories:

- `.:/app` - Main application code
- `./data:/app/data` - Data directory
- `./models:/app/models` - Model files

## Building for Different Architectures

While this Dockerfile is optimized for amd64, you can build for other architectures:

```bash
# Build for multiple architectures (requires buildx)
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 -t swarms:multiarch .
```

## Health Checks

The container includes a built-in health check that verifies the swarms package can be imported:

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' swarms-container
```

## Troubleshooting

### Issue: Architecture mismatch
If you're getting architecture-related errors, ensure Docker is configured for amd64:

```bash
# Check current architecture
docker info | grep Architecture

# Force amd64 platform
docker run --platform linux/amd64 swarms:latest
```

### Issue: Build fails
If the build fails, try cleaning Docker cache:

```bash
docker system prune -a
docker-compose build --no-cache
```

### Issue: Import errors
If you encounter import errors, rebuild the image:

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Advanced Usage

### Running Custom Scripts

```bash
# Run a Python script
docker run --platform linux/amd64 -v $(pwd):/app swarms:latest python your_script.py

# Using docker-compose
docker-compose run swarms python your_script.py
```

### Interactive Python Shell

```bash
# Start Python REPL
docker run --platform linux/amd64 -it swarms:latest python

# Or with docker-compose
docker-compose run swarms python
```

### Installing Additional Dependencies

Create a custom Dockerfile extending the base image:

```dockerfile
FROM swarms:latest

USER root
RUN pip install additional-package
USER swarms
```

## Production Deployment

For production deployments:

1. Use specific version tags instead of `latest`
2. Configure proper logging and monitoring
3. Set resource limits in docker-compose.yml
4. Use secrets management for API keys
5. Enable automatic container restarts

Example production configuration:

```yaml
services:
  swarms:
    image: swarms:1.0.0
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/kyegomez/swarms/issues
- Documentation: https://docs.swarms.world

## License

MIT License - See LICENSE file for details

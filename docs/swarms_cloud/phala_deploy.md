# 🔐 Swarms x Phala Deployment Guide

This guide will walk you through deploying your project to Phala's Trusted Execution Environment (TEE).

## 📋 Prerequisites

- Docker installed on your system
- A DockerHub account
- Access to Phala Cloud dashboard

## 🛡️ TEE Overview

For detailed instructions about Trusted Execution Environment setup, please refer to our [TEE Documentation](./tee/README.md).

## 🚀 Deployment Steps

### 1. Build and Publish Docker Image

```bash
# Build the Docker image
docker compose build -t <your-dockerhub-username>/swarm-agent-node:latest

# Push to DockerHub
docker push <your-dockerhub-username>/swarm-agent-node:latest
```

### 2. Deploy to Phala Cloud

Choose one of these deployment methods:
- Use [tee-cloud-cli](https://github.com/Phala-Network/tee-cloud-cli) (Recommended)
- Deploy manually via the [Phala Cloud Dashboard](https://cloud.phala.network/)

### 3. Verify TEE Attestation

Visit the [TEE Attestation Explorer](https://proof.t16z.com/) to check and verify your agent's TEE proof.

## 📝 Docker Configuration

Below is a sample Docker Compose configuration for your Swarms agent:

```yaml
services:
  swarms-agent-server:
    image: swarms-agent-node:latest
    platform: linux/amd64
    volumes:
      - /var/run/tappd.sock:/var/run/tappd.sock
      - swarms:/app
    restart: always
    ports:
      - 8000:8000
    command: # Sample MCP Server
      - /bin/sh
      - -c
      - |
        cd /app/mcp_example
        python mcp_test.py
volumes:
  swarms:
```

## 📚 Additional Resources

For more comprehensive documentation and examples, visit our [Official Documentation](https://docs.swarms.world/en/latest/).

---

> **Note**: Make sure to replace `<your-dockerhub-username>` with your actual DockerHub username when building and pushing the image.
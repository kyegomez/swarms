# Deploying Swarms on Phala Network

## Overview

!!! abstract "What you'll learn"
    Deploy Swarms agents on Phala Network's decentralized cloud infrastructure, leveraging secure and private execution environments through Trusted Execution Environments (TEE).

## Prerequisites

!!! check "Before you begin"
    Make sure you have:

    
    - [:simple-docker: Docker](https://www.docker.com/get-started){ .md-button } installed on your system
    
    - [:simple-docker: DockerHub](https://hub.docker.com/){ .md-button } account
    
    - [:octicons-terminal-24: tee-cloud-cli](https://github.com/Phala-Network/tee-cloud-cli){ .md-button } installed
    
    - [:simple-phala: Phala Network](https://phala.network){ .md-button } account with PHA tokens

## Quick Start Guide

=== "1. Build Docker Image"
    ```bash
    docker compose build -t <your-dockerhub-username>/swarm-agent-node:latest
    ```

=== "2. Push to DockerHub"
    ```bash
    docker push <your-dockerhub-username>/swarm-agent-node:latest
    ```

=== "3. Deploy to Phala"
    ```bash
    tee-cloud-cli deploy --image <your-dockerhub-username>/swarm-agent-node:latest
    ```

## Deployment Methods

### Using tee-cloud-cli (Recommended)

!!! tip "Recommended Method"
    This is the preferred way to deploy your Swarms agents.

1. Install the CLI:
    ```bash
    npm install -g @phala/tee-cloud-cli
    ```

2. Configure credentials:
    ```bash
    tee-cloud-cli config set --key YOUR_KEY
    ```

3. Deploy your agent:
    ```bash
    tee-cloud-cli deploy --image your-image --name your-agent-name
    ```

### Using Phala Dashboard


1. Visit [:material-web: Phala Cloud Dashboard](https://cloud.phala.network/)

2. Connect your wallet

3. Navigate to "Deploy" section

4. Configure your deployment

5. Click "Deploy"

## Configuration

### Environment Variables

!!! example "Configuration Example"
    ```yaml
    SWARMS_API_KEY: "your-api-key"
    PHALA_ENDPOINT: "wss://api.phala.network/ws"
    TEE_WORKER_ENDPOINT: "http://localhost:8000"
    ```

### Resource Requirements

| Resource | Recommended |
|----------|------------|
| CPU      | 2-4 cores  |
| Memory   | 4-8 GB     |
| Storage  | 20-50 GB   |

## Monitoring

### Viewing Logs

=== "Using Dashboard"
    Access logs through the Phala Dashboard interface

=== "Using CLI"
    ```bash
    tee-cloud-cli logs --name your-agent-name
    ```

### Health Checks

```bash
tee-cloud-cli status --name your-agent-name
```

## Best Practices

!!! success "Recommended Practices"
    1. **Version Control**: Tag Docker images with specific versions
    
    2. **Security**: Keep credentials out of Docker images
    
    3. **Resource Management**: Monitor and adjust resource usage
    
    4. **Backup**: Maintain regular backups
    
    5. **Testing**: Use staging environment first

## Troubleshooting

??? question "Deployment Failures"
    
    - Verify Docker image accessibility
    
    - Check resource allocations
    
    - Validate environment variables

??? question "Connection Issues"
    
    - Verify network connectivity
    
    - Check Phala endpoint status
    
    - Validate API credentials

??? question "Performance Problems"
    
    - Monitor resource usage
    
    - Adjust allocation if needed
    
    - Check for memory leaks

## FAQ

??? question "What is TEE?"
    TEE (Trusted Execution Environment) is a secure area of a processor that guarantees code and data loaded inside is protected with respect to confidentiality and integrity.

??? question "How much does it cost to deploy?"
    Deployment costs vary based on resource usage and current PHA token prices. Check the [Phala pricing calculator](https://cloud.phala.network) for current rates.

??? question "Can I use custom Docker images?"
    Yes, you can use any Docker image that's compatible with the Phala Network requirements and accessible on DockerHub.

??? question "How do I scale my deployment?"
    You can scale by adjusting resource allocations or deploying multiple instances through the dashboard or CLI.

## Support & Community

Need help? Join our communities:

[:simple-discord: Phala Discord](https://discord.gg/phala){ .md-button }
[:simple-discord: Swarms Discord](https://discord.gg/swarms){ .md-button }
[:material-book: Phala Docs](https://docs.phala.network){ .md-button }
[:octicons-mark-github-16: GitHub Issues](https://github.com/kyegomez/swarms/issues){ .md-button }
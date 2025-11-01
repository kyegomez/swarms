#!/bin/bash

# Quick start script for Swarms Docker container (x86_64/amd64)

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Swarms Docker Quick Start (amd64)"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if image exists
if ! docker image inspect swarms:latest &> /dev/null; then
    echo -e "${YELLOW}Swarms image not found. Building image...${NC}"
    ./build-docker-amd64.sh
fi

echo -e "${BLUE}Starting Swarms container...${NC}"

# Check if using docker-compose or docker run
if [ "$1" == "compose" ]; then
    echo -e "${BLUE}Using docker-compose...${NC}"
    docker-compose up -d
    echo -e "${GREEN}Container started! View logs with: docker-compose logs -f${NC}"
else
    echo -e "${BLUE}Using docker run...${NC}"
    docker run --platform linux/amd64 \
        --name swarms-container \
        -v $(pwd):/app \
        -e PYTHONUNBUFFERED=1 \
        -it \
        --rm \
        swarms:latest bash
fi

echo -e "${GREEN}=========================================="
echo "Swarms is ready!"
echo "==========================================${NC}"

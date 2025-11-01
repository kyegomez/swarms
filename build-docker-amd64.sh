#!/bin/bash

# Build script for Swarms Docker image (x86_64/amd64)
# This script builds the Swarms Docker image specifically for amd64 architecture

set -e

echo "=========================================="
echo "Building Swarms Docker Image for amd64"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check current architecture
ARCH=$(uname -m)
echo -e "${BLUE}Current system architecture: $ARCH${NC}"

# Build image
echo -e "${BLUE}Building Docker image for linux/amd64...${NC}"
docker build --platform linux/amd64 -t swarms:amd64 -t swarms:latest .

# Verify the build
echo -e "${BLUE}Verifying the build...${NC}"
docker images swarms

echo -e "${GREEN}=========================================="
echo "Build completed successfully!"
echo "==========================================${NC}"
echo ""
echo "To run the container:"
echo "  docker run --platform linux/amd64 -it swarms:latest bash"
echo ""
echo "Or use docker-compose:"
echo "  docker-compose up -d"
echo ""
echo "For more information, see DOCKER.md"

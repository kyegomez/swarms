#!/bin/bash

# Docker utilities for Swarms project
# Usage: ./scripts/docker-utils.sh [command]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="swarms"
REGISTRY="kyegomez"
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}"

# Functions
print_usage() {
    echo -e "${BLUE}Docker Utilities for Swarms${NC}"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build          Build the Docker image locally"
    echo "  test           Test the Docker image"
    echo "  run            Run the Docker image interactively"
    echo "  push           Push to DockerHub (requires login)"
    echo "  clean          Clean up Docker images and containers"
    echo "  logs           Show logs from running containers"
    echo "  shell          Open shell in running container"
    echo "  compose-up     Start services with docker-compose"
    echo "  compose-down   Stop services with docker-compose"
    echo "  help           Show this help message"
    echo ""
}

build_image() {
    echo -e "${GREEN}Building Docker image...${NC}"
    docker build -t "${IMAGE_NAME}:latest" .
    echo -e "${GREEN} Image built successfully!${NC}"
}

test_image() {
    echo -e "${GREEN}Testing Docker image...${NC}"
    docker run --rm "${IMAGE_NAME}:latest" python test_docker.py
    echo -e "${GREEN} Image test completed!${NC}"
}

run_interactive() {
    echo -e "${GREEN}Running Docker image interactively...${NC}"
    docker run -it --rm \
        -v "$(pwd):/app" \
        -w /app \
        "${IMAGE_NAME}:latest" bash
}

push_to_dockerhub() {
    echo -e "${YELLOW}⚠  Make sure you're logged into DockerHub first!${NC}"
    echo -e "${GREEN}Pushing to DockerHub...${NC}"
    
    # Tag the image
    docker tag "${IMAGE_NAME}:latest" "${FULL_IMAGE_NAME}:latest"
    
    # Push to DockerHub
    docker push "${FULL_IMAGE_NAME}:latest"
    
    echo -e "${GREEN} Image pushed to DockerHub!${NC}"
}

clean_docker() {
    echo -e "${YELLOW}Cleaning up Docker resources...${NC}"
    
    # Stop and remove containers
    docker ps -aq | xargs -r docker rm -f
    
    # Remove images
    docker images "${IMAGE_NAME}" -q | xargs -r docker rmi -f
    
    # Remove dangling images
    docker image prune -f
    
    echo -e "${GREEN} Docker cleanup completed!${NC}"
}

show_logs() {
    echo -e "${GREEN}Showing logs from running containers...${NC}"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    
    # Show logs for swarms containers
    for container in $(docker ps --filter "name=swarms" --format "{{.Names}}"); do
        echo -e "${BLUE}Logs for $container:${NC}"
        docker logs "$container" --tail 20
        echo ""
    done
}

open_shell() {
    echo -e "${GREEN}Opening shell in running container...${NC}"
    
    # Find running swarms container
    container=$(docker ps --filter "name=swarms" --format "{{.Names}}" | head -1)
    
    if [ -z "$container" ]; then
        echo -e "${RED} No running swarms container found!${NC}"
        echo "Start a container first with: $0 run"
        exit 1
    fi
    
    echo -e "${BLUE}Opening shell in $container...${NC}"
    docker exec -it "$container" bash
}

compose_up() {
    echo -e "${GREEN}Starting services with docker-compose...${NC}"
    docker-compose up -d
    echo -e "${GREEN} Services started!${NC}"
    echo "Use 'docker-compose logs -f' to view logs"
}

compose_down() {
    echo -e "${YELLOW}Stopping services with docker-compose...${NC}"
    docker-compose down
    echo -e "${GREEN} Services stopped!${NC}"
}

# Main script logic
case "${1:-help}" in
    build)
        build_image
        ;;
    test)
        test_image
        ;;
    run)
        run_interactive
        ;;
    push)
        push_to_dockerhub
        ;;
    clean)
        clean_docker
        ;;
    logs)
        show_logs
        ;;
    shell)
        open_shell
        ;;
    compose-up)
        compose_up
        ;;
    compose-down)
        compose_down
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED} Unknown command: $1${NC}"
        print_usage
        exit 1
        ;;
esac 

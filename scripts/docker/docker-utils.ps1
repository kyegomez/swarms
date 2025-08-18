# Docker utilities for Swarms project (PowerShell version)
# Usage: .\scripts\docker-utils.ps1 [command]

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

# Configuration
$ImageName = "swarms"
$Registry = "kyegomez"
$FullImageName = "$Registry/$ImageName"

# Functions
function Write-Usage {
    Write-Host "Docker Utilities for Swarms" -ForegroundColor Blue
    Write-Host ""
    Write-Host "Usage: .\scripts\docker-utils.ps1 [command]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  build          Build the Docker image locally"
    Write-Host "  test           Test the Docker image"
    Write-Host "  run            Run the Docker image interactively"
    Write-Host "  push           Push to DockerHub (requires login)"
    Write-Host "  clean          Clean up Docker images and containers"
    Write-Host "  logs           Show logs from running containers"
    Write-Host "  shell          Open shell in running container"
    Write-Host "  compose-up     Start services with docker-compose"
    Write-Host "  compose-down   Stop services with docker-compose"
    Write-Host "  help           Show this help message"
    Write-Host ""
}

function Build-Image {
    Write-Host "Building Docker image..." -ForegroundColor Green
    docker build -t "$ImageName`:latest" .
    Write-Host " Image built successfully!" -ForegroundColor Green
}

function Test-Image {
    Write-Host "Testing Docker image..." -ForegroundColor Green
    docker run --rm "$ImageName`:latest" python test_docker.py
    Write-Host " Image test completed!" -ForegroundColor Green
}

function Run-Interactive {
    Write-Host "Running Docker image interactively..." -ForegroundColor Green
    docker run -it --rm -v "${PWD}:/app" -w /app "$ImageName`:latest" bash
}

function Push-ToDockerHub {
    Write-Host "⚠  Make sure you're logged into DockerHub first!" -ForegroundColor Yellow
    Write-Host "Pushing to DockerHub..." -ForegroundColor Green
    
    # Tag the image
    docker tag "$ImageName`:latest" "$FullImageName`:latest"
    
    # Push to DockerHub
    docker push "$FullImageName`:latest"
    
    Write-Host " Image pushed to DockerHub!" -ForegroundColor Green
}

function Clean-Docker {
    Write-Host "Cleaning up Docker resources..." -ForegroundColor Yellow
    
    # Stop and remove containers
    docker ps -aq | ForEach-Object { docker rm -f $_ }
    
    # Remove images
    docker images "$ImageName" -q | ForEach-Object { docker rmi -f $_ }
    
    # Remove dangling images
    docker image prune -f
    
    Write-Host " Docker cleanup completed!" -ForegroundColor Green
}

function Show-Logs {
    Write-Host "Showing logs from running containers..." -ForegroundColor Green
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    Write-Host ""
    
    # Show logs for swarms containers
    $containers = docker ps --filter "name=swarms" --format "{{.Names}}"
    foreach ($container in $containers) {
        Write-Host "Logs for $container:" -ForegroundColor Blue
        docker logs $container --tail 20
        Write-Host ""
    }
}

function Open-Shell {
    Write-Host "Opening shell in running container..." -ForegroundColor Green
    
    # Find running swarms container
    $container = docker ps --filter "name=swarms" --format "{{.Names}}" | Select-Object -First 1
    
    if (-not $container) {
        Write-Host " No running swarms container found!" -ForegroundColor Red
        Write-Host "Start a container first with: .\scripts\docker-utils.ps1 run"
        exit 1
    }
    
    Write-Host "Opening shell in $container..." -ForegroundColor Blue
    docker exec -it $container bash
}

function Compose-Up {
    Write-Host "Starting services with docker-compose..." -ForegroundColor Green
    docker-compose up -d
    Write-Host " Services started!" -ForegroundColor Green
    Write-Host "Use 'docker-compose logs -f' to view logs"
}

function Compose-Down {
    Write-Host "Stopping services with docker-compose..." -ForegroundColor Yellow
    docker-compose down
    Write-Host " Services stopped!" -ForegroundColor Green
}

# Main script logic
switch ($Command.ToLower()) {
    "build" { Build-Image }
    "test" { Test-Image }
    "run" { Run-Interactive }
    "push" { Push-ToDockerHub }
    "clean" { Clean-Docker }
    "logs" { Show-Logs }
    "shell" { Open-Shell }
    "compose-up" { Compose-Up }
    "compose-down" { Compose-Down }
    "help" { Write-Usage }
    default {
        Write-Host " Unknown command: $Command" -ForegroundColor Red
        Write-Usage
        exit 1
    }
} 

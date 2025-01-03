"""
SkyServe API: Production-grade FastAPI server for SimpleSkyServe.

This module provides a REST API interface for managing SkyPilot services with 
proper error handling, validation, and production configurations.
"""

import multiprocessing
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field
from pydantic.v1 import validator
from swarm_cloud_code import ServiceConfig, SimpleSkyServe, UpdateMode

# Calculate optimal number of workers
CPU_COUNT = multiprocessing.cpu_count()
WORKERS = CPU_COUNT * 2

# Configure logging
logger.add(
    "logs/skyserve-api.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
)

# Initialize FastAPI app
app = FastAPI(
    title="SkyServe API",
    description="REST API for managing SkyPilot services",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response validation
class ServiceConfigRequest(BaseModel):
    """Request model for service configuration."""

    code: str = Field(
        ..., description="Python code to run as a service"
    )
    requirements: Optional[List[str]] = Field(
        default=None, description="List of pip packages"
    )
    envs: Optional[dict] = Field(
        default=None, description="Environment variables"
    )
    name: Optional[str] = Field(
        default=None, description="Service name"
    )
    num_cpus: int = Field(
        default=2, ge=1, description="Number of CPUs"
    )
    memory: int = Field(default=4, ge=1, description="Memory in GB")
    use_spot: bool = Field(
        default=False, description="Use spot instances"
    )
    num_nodes: int = Field(
        default=1, ge=1, description="Number of nodes"
    )

    @validator("name")
    def validate_name(cls, v):
        if v and not v.isalnum():
            raise ValueError("Service name must be alphanumeric")
        return v


class DeploymentResponse(BaseModel):
    """Response model for deployment information."""

    service_name: str
    endpoint: str


class ServiceStatusResponse(BaseModel):
    """Response model for service status."""

    name: str
    status: str
    versions: List[int]
    replicas: int
    resources: str
    uptime: int
    endpoint: Optional[str]


@app.post(
    "/services/",
    response_model=DeploymentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["services"],
)
async def create_service(config: ServiceConfigRequest):
    """Deploy a new service."""
    try:
        service_config = ServiceConfig(
            code=config.code,
            requirements=config.requirements,
            envs=config.envs,
            name=config.name,
            num_cpus=config.num_cpus,
            memory=config.memory,
            use_spot=config.use_spot,
            num_nodes=config.num_nodes,
        )
        name, endpoint = SimpleSkyServe.deploy(service_config)
        return {"service_name": name, "endpoint": endpoint}
    except Exception as e:
        logger.error(f"Failed to create service: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get(
    "/services/",
    response_model=List[ServiceStatusResponse],
    tags=["services"],
)
async def list_services(name: Optional[str] = None):
    """Get status of all services or a specific service."""
    try:
        deployments = SimpleSkyServe.get_deployments(name)
        return deployments
    except Exception as e:
        logger.error(f"Failed to list services: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.put(
    "/services/{service_name}",
    status_code=status.HTTP_200_OK,
    tags=["services"],
)
async def update_service(
    service_name: str,
    config: ServiceConfigRequest,
    mode: UpdateMode = UpdateMode.GRADUAL,
):
    """Update an existing service."""
    try:
        service_config = ServiceConfig(
            code=config.code,
            requirements=config.requirements,
            envs=config.envs,
            name=config.name,
            num_cpus=config.num_cpus,
            memory=config.memory,
            use_spot=config.use_spot,
            num_nodes=config.num_nodes,
        )
        SimpleSkyServe.update(service_name, service_config, mode)
        return {
            "message": f"Service {service_name} updated successfully"
        }
    except Exception as e:
        logger.error(f"Failed to update service: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.delete(
    "/services/{service_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["services"],
)
async def delete_service(service_name: str, purge: bool = False):
    """Delete a service."""
    try:
        SimpleSkyServe.delete(service_name, purge)
    except Exception as e:
        logger.error(f"Failed to delete service: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# Entry point for uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        workers=WORKERS,
        log_level="info",
        reload=False,  # Disable in production
        proxy_headers=True,
        forwarded_allow_ips="*",
        access_log=True,
    )

"""
SimpleSkyServe: A simplified interface for SkyPilot's serve functionality.

This module provides an easy-to-use interface for deploying, managing, updating and monitoring
services using SkyPilot's serve functionality. It supports the full lifecycle of services
including deployment, updates, status monitoring, and cleanup.

Key Features:
- Simple deployment with code and requirements
- Service updates with different update modes 
- Status monitoring and deployment fetching
- Service cleanup and deletion
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import tempfile

from loguru import logger


class UpdateMode(Enum):
    """Update modes for service updates.

    IMMEDIATE: Update all replicas immediately
    GRADUAL: Update replicas gradually with zero downtime
    """

    IMMEDIATE = "immediate"
    GRADUAL = "gradual"


@dataclass
class ServiceConfig:
    """Configuration for a SkyPilot service.

    Attributes:
        code: Python code to run as a service
        requirements: List of pip packages required by the service
        envs: Environment variables to set for the service
        name: Optional name for the service (auto-generated if not provided)
        num_cpus: Number of CPUs to request (default: 2)
        memory: Memory in GB to request (default: 4)
        use_spot: Whether to use spot instances (default: False)
    """

    code: str
    requirements: Optional[List[str]] = None
    envs: Optional[Dict[str, str]] = None
    name: Optional[str] = None
    num_cpus: int = 2
    memory: int = 4
    use_spot: bool = False
    num_nodes: int = 1


class SimpleSkyServe:
    """Simple interface for SkyPilot serve functionality."""

    @staticmethod
    def deploy(config: ServiceConfig) -> Tuple[str, str]:
        """Deploy a new service using the provided configuration.

        Args:
            config: ServiceConfig object containing service configuration

        Returns:
            Tuple of (service_name: str, endpoint: str)

        Raises:
            ValueError: If the configuration is invalid
            RuntimeError: If deployment fails
        """
        logger.info("Deploying new service...")

        # Create temporary files for setup and service code
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt"
        ) as req_file, tempfile.NamedTemporaryFile(
            mode="w", suffix=".py"
        ) as code_file:

            # Write requirements if provided
            setup_cmd = ""
            if config.requirements:
                req_file.write("\n".join(config.requirements))
                req_file.flush()
                setup_cmd = f"pip install -r {req_file.name}"

            # Write service code
            code_file.write(config.code)
            code_file.flush()

            # Create SkyPilot task
            task = sky.Task(
                name=config.name,
                setup=setup_cmd,
                run=f"python {code_file.name}",
                envs=config.envs,
                num_nodes=config.num_nodes,
            )

            # Set resource requirements
            resources = sky.Resources(
                cpus=config.num_cpus,
                memory=config.memory,
                use_spot=config.use_spot,
            )
            task.set_resources(resources)

            try:
                # Deploy the service
                service_name, endpoint = sky.serve.up(
                    task, service_name=config.name
                )
                logger.success(
                    f"Service deployed successfully at {endpoint}"
                )
                return service_name, endpoint
            except Exception as e:
                logger.error(f"Failed to deploy service: {str(e)}")
                raise RuntimeError(
                    f"Service deployment failed: {str(e)}"
                ) from e

    @staticmethod
    def status(service_name: Optional[str] = None) -> List[Dict]:
        """Get status of services.

        Args:
            service_name: Optional name of specific service to get status for
                        If None, returns status of all services

        Returns:
            List of service status dictionaries containing:
                - name: Service name
                - status: Current status
                - endpoint: Service endpoint
                - uptime: Service uptime in seconds
                ...and other service metadata
        """
        logger.info(
            f"Getting status for service: {service_name or 'all'}"
        )
        try:
            status_list = sky.serve.status(service_name)
            logger.debug(
                f"Retrieved status for {len(status_list)} services"
            )
            return status_list
        except Exception as e:
            logger.error(f"Failed to get service status: {str(e)}")
            raise RuntimeError(
                f"Failed to get service status: {str(e)}"
            ) from e

    @staticmethod
    def update(
        service_name: str,
        config: ServiceConfig,
        mode: UpdateMode = UpdateMode.GRADUAL,
    ) -> None:
        """Update an existing service with new configuration.

        Args:
            service_name: Name of service to update
            config: New service configuration
            mode: Update mode (IMMEDIATE or GRADUAL)

        Raises:
            ValueError: If service doesn't exist or config is invalid
            RuntimeError: If update fails
        """
        logger.info(
            f"Updating service {service_name} with mode {mode.value}"
        )

        # Create temporary files for setup and service code
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt"
        ) as req_file, tempfile.NamedTemporaryFile(
            mode="w", suffix=".py"
        ) as code_file:

            # Write requirements if provided
            setup_cmd = ""
            if config.requirements:
                req_file.write("\n".join(config.requirements))
                req_file.flush()
                setup_cmd = f"pip install -r {req_file.name}"

            # Write service code
            code_file.write(config.code)
            code_file.flush()

            # Create SkyPilot task for update
            task = sky.Task(
                name=config.name or service_name,
                setup=setup_cmd,
                run=f"python {code_file.name}",
                envs=config.envs,
            )

            # Set resource requirements
            resources = sky.Resources(
                cpus=config.num_cpus,
                memory=config.memory,
                use_spot=config.use_spot,
            )
            task.set_resources(resources)

            try:
                # Update the service
                sky.serve.update(
                    task=task,
                    service_name=service_name,
                    mode=sky.serve.UpdateMode(mode.value),
                )
                logger.success(
                    f"Service {service_name} updated successfully"
                )
            except Exception as e:
                logger.error(f"Failed to update service: {str(e)}")
                raise RuntimeError(
                    f"Service update failed: {str(e)}"
                ) from e

    @staticmethod
    def get_deployments(
        service_name: Optional[str] = None,
    ) -> List[Dict]:
        """Get detailed information about service deployments.

        Args:
            service_name: Optional name of specific service to get deployments for
                        If None, returns deployments for all services

        Returns:
            List of deployment dictionaries containing:
                - name: Service name
                - versions: List of deployed versions
                - active_version: Currently active version
                - replicas: Number of replicas
                - resources: Resource usage
                - status: Deployment status
        """
        logger.info(
            f"Fetching deployments for: {service_name or 'all services'}"
        )
        try:
            status_list = sky.serve.status(service_name)
            deployments = []

            for status in status_list:
                deployment = {
                    "name": status["name"],
                    "versions": status["active_versions"],
                    "status": status["status"],
                    "replicas": len(status.get("replica_info", [])),
                    "resources": status.get(
                        "requested_resources_str", ""
                    ),
                    "uptime": status.get("uptime", 0),
                    "endpoint": None,
                }

                # Extract endpoint if available
                if status.get("load_balancer_port"):
                    deployment["endpoint"] = (
                        f"http://{status.get('controller_addr')}:{status['load_balancer_port']}"
                    )

                deployments.append(deployment)

            logger.debug(f"Retrieved {len(deployments)} deployments")
            return deployments

        except Exception as e:
            logger.error(f"Failed to fetch deployments: {str(e)}")
            raise RuntimeError(
                f"Failed to fetch deployments: {str(e)}"
            ) from e

    @staticmethod
    def delete(
        service_name: Union[str, List[str]], purge: bool = False
    ) -> None:
        """Delete one or more services.

        Args:
            service_name: Name of service(s) to delete
            purge: Whether to purge services in failed status

        Raises:
            RuntimeError: If deletion fails
        """
        names = (
            [service_name]
            if isinstance(service_name, str)
            else service_name
        )
        logger.info(f"Deleting services: {names}")
        try:
            sky.serve.down(service_names=names, purge=purge)
            logger.success(f"Successfully deleted services: {names}")
        except Exception as e:
            logger.error(f"Failed to delete services: {str(e)}")
            raise RuntimeError(
                f"Service deletion failed: {str(e)}"
            ) from e


# # Example usage:
# if __name__ == "__main__":
#     from time import sleep
#     # Configuration for a simple FastAPI service
#     config = ServiceConfig(
#         code="""
# from fastapi import FastAPI
# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}
#         """,
#         requirements=["fastapi", "uvicorn"],
#         envs={"PORT": "8000"},
#         name="fastapi-demo"
#     )

#     # Deploy the service
#     name, endpoint = SimpleSkyServe.deploy(config)
#     print(f"Service deployed at: {endpoint}")

#     # Get service status
#     status = SimpleSkyServe.status(name)
#     print(f"Service status: {status}")

#     # Get deployment information
#     deployments = SimpleSkyServe.get_deployments(name)
#     print(f"Deployment info: {deployments}")

#     # Update the service with new code
#     new_config = ServiceConfig(
#         code="""
# from fastapi import FastAPI
# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"Hello": "Updated World"}
#         """,
#         requirements=["fastapi", "uvicorn"],
#         envs={"PORT": "8000"}
#     )

#     SimpleSkyServe.update(name, new_config, mode=UpdateMode.GRADUAL)
#     print("Service updated")

#     # Wait for update to complete
#     sleep(30)

#     # Check status after update
#     status = SimpleSkyServe.status(name)
#     print(f"Updated service status: {status}")

#     # Delete the service
#     SimpleSkyServe.delete(name)

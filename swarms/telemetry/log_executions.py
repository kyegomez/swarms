from typing import Optional
from swarms.telemetry.main import log_agent_data


def log_execution(
    swarm_id: Optional[str] = None,
    status: Optional[str] = None,
    swarm_config: Optional[dict] = None,
    swarm_architecture: Optional[str] = None,
):
    """
    Log execution data for a swarm router instance.

    This function logs telemetry data about swarm router executions, including
    the swarm ID, execution status, and configuration details. It silently
    handles any logging errors to prevent execution interruption.

    Args:
        swarm_id (str): Unique identifier for the swarm router instance
        status (str): Current status of the execution (e.g., "start", "completion", "error")
        swarm_config (dict): Configuration dictionary containing swarm router settings
        swarm_architecture (str): Name of the swarm architecture used
    Returns:
        None

    Example:
        >>> log_execution(
        ...     swarm_id="swarm-router-abc123",
        ...     status="start",
        ...     swarm_config={"name": "my-swarm", "swarm_type": "SequentialWorkflow"}
        ... )
    """
    try:
        log_agent_data(
            data_dict={
                "swarm_router_id": swarm_id,
                "status": status,
                "swarm_router_config": swarm_config,
                "swarm_architecture": swarm_architecture,
            }
        )
    except Exception:
        pass

from typing import Any, Callable, Dict, Optional, Sequence

from swarms.structs.base_swarm import BaseSwarm
from swarms.utils.loguru_logger import logger


class AutoSwarmRouter(BaseSwarm):
    """AutoSwarmRouter class represents a router for the AutoSwarm class.

    This class is responsible for routing tasks to the appropriate swarm based on the provided name.
    It allows customization of the preprocessing, routing, and postprocessing of tasks.

    Attributes:
        name (str): The name of the router.
        description (str): The description of the router.
        verbose (bool): Whether to enable verbose mode.
        custom_params (dict): Custom parameters for the router.
        swarms (list): A list of BaseSwarm objects.
        custom_preprocess (callable): Custom preprocessing function for tasks.
        custom_postprocess (callable): Custom postprocessing function for task results.
        custom_router (callable): Custom routing function for tasks.

    Methods:
        run(task: str = None, *args, **kwargs) -> Any:
            Run the swarm simulation and route the task to the appropriate swarm.

    Flow:
        name -> router -> swarm entry point
    """

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        verbose: bool = False,
        custom_params: Optional[Dict[str, Any]] = None,
        swarms: Sequence[BaseSwarm] = None,
        custom_preprocess: Optional[Callable] = None,
        custom_postprocess: Optional[Callable] = None,
        custom_router: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.description = description
        self.verbose = verbose
        self.custom_params = custom_params
        self.swarms = swarms
        self.custom_preprocess = custom_preprocess
        self.custom_postprocess = custom_postprocess
        self.custom_router = custom_router

        # Create a dictionary of swarms
        self.swarm_dict = {swarm.name: swarm for swarm in self.swarms}

    def run(self, task: str = None, *args, **kwargs):
        try:
            """Run the swarm simulation and route the task to the appropriate swarm."""

            if self.custom_preprocess:
                # If custom preprocess function is provided then run it
                logger.info("Running custom preprocess function.")
                task, args, kwargs = self.custom_preprocess(
                    task, args, kwargs
                )

            if self.custom_router:
                # If custom router function is provided then use it to route the task
                logger.info("Running custom router function.")
                out = self.custom_router(self, task, *args, **kwargs)

                if self.custom_postprocess:
                    # If custom postprocess function is provided then run it
                    out = self.custom_postprocess(out)

                return out

            if self.name in self.swarm_dict:
                # If a match is found then send the task to the swarm
                out = self.swarm_dict[self.name].run(task, *args, **kwargs)

                if self.custom_postprocess:
                    # If custom postprocess function is provided then run it
                    out = self.custom_postprocess(out)

                return out

            # If no match is found then return None
            raise ValueError(f"Swarm with name {self.name} not found.")
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e


class AutoSwarm(BaseSwarm):
    """AutoSwarm class represents a swarm of agents that can be created automatically.

    Flow:
    name -> router -> swarm entry point

    Args:
        name (Optional[str]): The name of the swarm. Defaults to None.
        description (Optional[str]): The description of the swarm. Defaults to None.
        verbose (bool): Whether to enable verbose mode. Defaults to False.
        custom_params (Optional[Dict[str, Any]]): Custom parameters for the swarm. Defaults to None.
        router (Optional[AutoSwarmRouter]): The router for the swarm. Defaults to None.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        verbose: bool = False,
        custom_params: Optional[Dict[str, Any]] = None,
        custom_preprocess: Optional[Callable] = None,
        custom_postprocess: Optional[Callable] = None,
        custom_router: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.description = description
        self.verbose = verbose
        self.custom_params = custom_params
        self.custom_preprocess = custom_preprocess
        self.custom_postprocess = custom_postprocess
        self.router = AutoSwarmRouter(
            name=name,
            description=description,
            verbose=verbose,
            custom_params=custom_params,
            custom_preprocess=custom_preprocess,
            custom_postprocess=custom_postprocess,
            custom_router=custom_router,
            *args,
            **kwargs,
        )

    def run(self, task: str = None, *args, **kwargs):
        """Run the swarm simulation."""
        try:
            return self.router.run(task, *args, **kwargs)
        except Exception as e:
            logger.error(
                f"Error: {e} try optimizing the inputs and try again."
            )
            raise e

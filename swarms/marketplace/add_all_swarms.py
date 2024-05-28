import os
import importlib.util
from typing import List, Type
from swarms import AutoSwarm, AutoSwarmRouter, BaseSwarm


def find_base_swarm_classes(
    folder_path: str = "prebuilt",
) -> List[Type[BaseSwarm]]:
    """
    Find and return a list of all classes that inherit from the BaseSwarm class
    within the specified folder path.

    Args:
        folder_path (str): The path to the folder containing the swarm classes.
            Defaults to "prebuilt".

    Returns:
        List[Type[BaseSwarm]]: A list of all classes that inherit from the BaseSwarm class.
    """
    base_swarm_classes: List[Type[BaseSwarm]] = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file == "__init__.py":
                module_path: str = os.path.join(root, file)
                spec = importlib.util.spec_from_file_location(
                    "module.name", module_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name, obj in module.__dict__.items():
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, BaseSwarm)
                        and obj is not BaseSwarm
                    ):
                        base_swarm_classes.append(obj)

    return base_swarm_classes


# Define the folder containing the prebuilt swarms
prebuilt_folder: str = "prebuilt"

# Find all BaseSwarm classes in the prebuilt folder
prebuilt_swarms: List[Type[BaseSwarm]] = find_base_swarm_classes(
    prebuilt_folder
)

# Add all swarms to the AutoSwarmRouter
router: AutoSwarmRouter = AutoSwarmRouter(swarms=prebuilt_swarms)

# Create an AutoSwarm instance
autoswarm: AutoSwarm = AutoSwarm(
    router=router,
)

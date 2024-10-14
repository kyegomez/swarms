from typing import Callable, List, Dict, Any
from swarms.structs.base_swarm import BaseSwarm
from loguru import logger

import time
import uuid


class SwarmArrangeInput:
    id: str = uuid.uuid4().hex
    time_stamp: str = time.strftime("%Y-%m-%d %H:%M:%S")
    name: str
    description: str
    swarms: List[Callable] = []
    output_type: str
    flow: str = ""


class SwarmArrangeOutput:
    input_config: SwarmArrangeInput = None


class SwarmArrange:
    """
    A class for arranging and executing multiple swarms sequentially.

    Attributes:
        name (str): The name of the SwarmArrange instance.
        description (str): A description of the SwarmArrange instance.
        swarms (List[Callable]): A list of swarms to be arranged and executed.
        output_type (str): The type of output expected from the SwarmArrange instance.
        flow (str): The flow pattern of the swarms to be executed.
    """

    def __init__(
        self,
        name: str = "SwarmArrange-01",
        description: str = "Combine multiple swarms and execute them sequentially",
        swarms: List[Any] = [],
        output_type: str = "json",
        flow: str = None,
    ):
        """
        Initializes the SwarmArrange instance.

        Args:
            name (str, optional): The name of the SwarmArrange instance. Defaults to "SwarmArrange-01".
            description (str, optional): A description of the SwarmArrange instance. Defaults to "Combine multiple swarms and execute them sequentially".
            swarms (List[Callable], optional): A list of swarms to be arranged and executed. Defaults to None.
            output_type (str, optional): The type of output expected from the SwarmArrange instance. Defaults to "json".
            flow (str, optional): The flow pattern of the swarms to be executed. Defaults to None.

        Raises:
            ValueError: If the name or description is None.
        """
        if not name:
            raise ValueError("Name cannot be None")
        if not description:
            raise ValueError("Description cannot be None")
        self.name = name
        self.description = description
        self.swarms = swarms
        self.output_type = output_type
        self.flow = flow

        self.reliability_check()

        # self.log = SwarmArrangeInput(
        #     name=name,
        #     description=description,
        #     swarms=swarms,
        #     output_type=output_type,
        #     flow=flow,
        # )

    def reliability_check(self):
        """
        Performs a reliability check on the SwarmArrange instance.

        This method checks if the swarms provided are valid and logs the results.
        """
        logger.info(
            f"Initializing the SwarmArrange with name: {self.name} and description: {self.description}"
        )

        if self.swarms is None:
            logger.warning(
                "No swarms detected. Please input a callable with a .run(task: str) method for reliable operation."
            )
        else:
            logger.info(
                "SwarmArrange initialized with swarms. Proceeding with reliability check."
            )

        # Additional logging for reliability check
        logger.info(
            "Checking if all swarms are callable or instances of BaseSwarm."
        )
        for swarm in self.swarms:
            if not callable(swarm) and not isinstance(
                swarm, BaseSwarm
            ):
                logger.error(
                    f"Swarm {swarm} is not a callable or an instance of BaseSwarm. This may cause reliability issues."
                )
                return False
        logger.info("All swarms are valid. SwarmArrange is reliable.")
        return True

    def set_custom_flow(self, flow: str):
        """
        Sets a custom flow pattern for the SwarmArrange instance.

        Args:
            flow (str): The custom flow pattern to be set.
        """
        self.flow = flow
        logger.info(f"Custom flow set: {flow}")

    def add_swarm(self, swarm: Callable):
        """
        Adds an swarm to the SwarmArrange instance.

        Args:
            swarm (swarm): The swarm to be added.
        """
        logger.info(f"Adding swarm {swarm.name} to the swarm.")
        self.swarms[swarm.name] = swarm

    def track_history(
        self,
        swarm_name: str,
        result: str,
    ):
        """
        Tracks the history of a swarm's execution.

        Args:
            swarm_name (str): The name of the swarm.
            result (str): The result of the swarm's execution.
        """
        self.swarm_history[swarm_name].append(result)

    def remove_swarm(self, swarm_name: str):
        """
        Removes an swarm from the SwarmArrange instance.

        Args:
            swarm_name (str): The name of the swarm to be removed.
        """
        del self.swarms[swarm_name]

    def add_swarms(self, swarms: List[Callable]):
        """
        Adds multiple swarms to the SwarmArrange instance.

        Args:
            swarms (List[swarm]): A list of swarm objects.
        """
        self.swarms.extend(swarms)

    def validate_flow(self):
        """
        Validates the flow pattern of the SwarmArrange instance.

        Raises:
            ValueError: If the flow pattern is incorrectly formatted or contains duplicate swarm names.

        Returns:
            bool: True if the flow pattern is valid.
        """
        if "->" not in self.flow:
            raise ValueError(
                "Flow must include '->' to denote the direction of the task."
            )

        swarms_in_flow = []

        # Split the flow into tasks
        tasks = self.flow.split("->")

        # For each task in the tasks
        for task in tasks:
            swarm_names = [name.strip() for name in task.split(",")]

            # Loop over the swarm names
            for swarm_name in swarm_names:
                if (
                    swarm_name not in self.swarms
                    and swarm_name != "H"
                ):
                    raise ValueError(
                        f"swarm '{swarm_name}' is not registered."
                    )
                swarms_in_flow.append(swarm_name)

        # Check for duplicate swarm names in the flow
        if len(set(swarms_in_flow)) != len(swarms_in_flow):
            raise ValueError(
                "Duplicate swarm names in the flow are not allowed."
            )

        logger.info("Flow is valid.")
        return True

    def run(
        self,
        task: str = None,
        img: str = None,
        custom_tasks: Dict[str, str] = None,
        *args,
        **kwargs,
    ):
        """
        Runs the SwarmArrange instance to rearrange and execute the swarms.

        Args:
            task (str, optional): The initial task to be processed. Defaults to None.
            img (str, optional): The image to be processed. Defaults to None.
            custom_tasks (Dict[str, str], optional): Custom tasks to be executed. Defaults to None.

        Returns:
            str: The final processed task.
        """
        try:
            if not self.validate_flow():
                return "Invalid flow configuration."

            tasks = self.flow.split("->")
            current_task = task

            # If custom_tasks have the swarms name and tasks then combine them
            if custom_tasks is not None:
                c_swarm_name, c_task = next(
                    iter(custom_tasks.items())
                )

                # Find the position of the custom swarm in the tasks list
                position = tasks.index(c_swarm_name)

                # If there is a previous swarm merge its task with the custom tasks
                if position > 0:
                    tasks[position - 1] += "->" + c_task
                else:
                    # If there is no previous swarm just insert the custom tasks
                    tasks.insert(position, c_task)

            # Set the loop counter
            loop_count = 0
            while loop_count < self.max_loops:
                for task in tasks:
                    is_last = task == tasks[-1]
                    swarm_names = [
                        name.strip() for name in task.split(",")
                    ]
                    if len(swarm_names) > 1:
                        # Parallel processing
                        logger.info(
                            f"Running swarms in parallel: {swarm_names}"
                        )
                        results = []
                        for swarm_name in swarm_names:
                            if swarm_name == "H":
                                # Human in the loop intervention
                                if (
                                    self.human_in_the_loop
                                    and self.custom_human_in_the_loop
                                ):
                                    current_task = (
                                        self.custom_human_in_the_loop(
                                            current_task
                                        )
                                    )
                                else:
                                    current_task = input(
                                        "Enter your response:"
                                    )
                            else:
                                swarm = self.swarms[swarm_name]
                                result = swarm.run(
                                    current_task,
                                    # img,
                                    # is_last,
                                    *args,
                                    **kwargs,
                                )
                                results.append(result)
                                self.output_schema.outputs.append(
                                    swarm.swarm_output
                                )

                        current_task = "; ".join(results)
                    else:
                        # Sequential processing
                        logger.info(
                            f"Running swarms sequentially: {swarm_names}"
                        )
                        swarm_name = swarm_names[0]
                        if swarm_name == "H":
                            # Human-in-the-loop intervention
                            if (
                                self.human_in_the_loop
                                and self.custom_human_in_the_loop
                            ):
                                current_task = (
                                    self.custom_human_in_the_loop(
                                        current_task
                                    )
                                )
                            else:
                                current_task = input(
                                    "Enter the next task: "
                                )
                        else:
                            swarm = self.swarms[swarm_name]
                            current_task = swarm.run(
                                current_task,
                                # img,
                                # is_last,
                                *args,
                                **kwargs,
                            )
                            self.output_schema.outputs.append(
                                swarm.swarm_output
                            )
                loop_count += 1

            # return current_task
            if self.return_json:
                return self.output_schema.model_dump_json(indent=4)
            else:
                return current_task

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return e

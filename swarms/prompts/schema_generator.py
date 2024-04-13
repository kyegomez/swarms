import json
from typing import List

from swarms.tools.tool import BaseTool

FINISH_NAME = "finish"


class SchemaGenerator:
    """A class for generating custom prompt strings.

    Does this based on constraints, commands, resources, and performance evaluations.

    Attributes:
        constraints (List[str]): A list of constraints.
        commands (List[BaseTool]): A list of commands.
        resources (List[str]): A list of resources.
        performance_evaluation (List[str]): A list of performance evaluations.
        response_format (dict): A dictionary of the response format.

    Examples:
        >>> schema_generator = SchemaGenerator()
        >>> schema_generator.add_constraint("No user assistance")
        >>> schema_generator.add_resource("Internet access for searches and information gathering.")
        >>> schema_generator.add_performance_evaluation("Continuously review and analyze your actions to ensure you are performing to the best of your abilities.")
        >>> prompt_string = schema_generator.generate_prompt_string()
        >>> print(prompt_string)
    """

    def __init__(self) -> None:
        """Initialize the SchemaGenerator object.

        Starts with empty lists of constraints, commands, resources,
        and performance evaluations.
        """
        self.constraints: List[str] = []
        self.commands: List[BaseTool] = []
        self.resources: List[str] = []
        self.performance_evaluation: List[str] = []
        self.response_format = {
            "thoughts": {
                "text": "thought",
                "reasoning": "reasoning",
                "plan": (
                    "- short bulleted\n- list that conveys\n-"
                    " long-term plan"
                ),
                "criticism": "constructive self-criticism",
                "speak": "thoughts summary to say to user",
            },
            "command": {
                "name": "command name",
                "args": {"arg name": "value"},
            },
        }

    def add_constraint(self, constraint: str) -> None:
        """
        Add a constraint to the constraints list.

        Args:
            constraint (str): The constraint to be added.
        """
        self.constraints.append(constraint)

    def add_tool(self, tool: BaseTool) -> None:
        self.commands.append(tool)

    def _generate_command_string(self, tool: BaseTool) -> str:
        output = f"{tool.name}: {tool.description}"
        output += f", args json schema: {json.dumps(tool.args)}"
        return output

    def add_resource(self, resource: str) -> None:
        """
        Add a resource to the resources list.

        Args:
            resource (str): The resource to be added.
        """
        self.resources.append(resource)

    def add_performance_evaluation(self, evaluation: str) -> None:
        """
        Add a performance evaluation item to the performance_evaluation list.

        Args:
            evaluation (str): The evaluation item to be added.
        """
        self.performance_evaluation.append(evaluation)

    def _generate_numbered_list(
        self, items: list, item_type: str = "list"
    ) -> str:
        """
        Generate a numbered list from given items based on the item_type.

        Args:
            items (list): A list of items to be numbered.
            item_type (str, optional): The type of items in the list.
                Defaults to 'list'.

        Returns:
            str: The formatted numbered list.
        """
        if item_type == "command":
            command_strings = [
                f"{i + 1}. {self._generate_command_string(item)}"
                for i, item in enumerate(items)
            ]
            finish_description = (
                "use this to signal that you have finished all your"
                " objectives"
            )
            finish_args = (
                '"response": "final response to let '
                'people know you have finished your objectives"'
            )
            finish_string = (
                f"{len(items) + 1}. {FINISH_NAME}: "
                f"{finish_description}, args: {finish_args}"
            )
            return "\n".join(command_strings + [finish_string])
        else:
            return "\n".join(
                f"{i + 1}. {item}" for i, item in enumerate(items)
            )

    def generate_prompt_string(self) -> str:
        """Generate a prompt string.

        Returns:
            str: The generated prompt string.
        """
        formatted_response_format = json.dumps(
            self.response_format, indent=4
        )
        prompt_string = (
            f"Constraints:\n{self._generate_numbered_list(self.constraints)}\n\nCommands:\n{self._generate_numbered_list(self.commands, item_type='command')}\n\nResources:\n{self._generate_numbered_list(self.resources)}\n\nPerformance"
            f" Evaluation:\n{self._generate_numbered_list(self.performance_evaluation)}\n\nYou"
            " should only respond in JSON format as described below"
            " \nResponse Format:"
            f" \n{formatted_response_format} \nEnsure the response"
            " can be parsed by Python json.loads"
        )

        return prompt_string


def get_prompt(tools: List[BaseTool]) -> str:
    """Generates a prompt string.

    It includes various constraints, commands, resources, and performance evaluations.

    Returns:
        str: The generated prompt string.
    """

    # Initialize the SchemaGenerator object
    schema_generator = SchemaGenerator()

    # Add constraints to the SchemaGenerator object
    schema_generator.add_constraint(
        "~4000 word limit for short term memory. "
        "Your short term memory is short, "
        "so immediately save important information to files."
    )
    schema_generator.add_constraint(
        "If you are unsure how you previously did something "
        "or want to recall past events, "
        "thinking about similar events will help you remember."
    )
    schema_generator.add_constraint("No user assistance")
    schema_generator.add_constraint(
        "Exclusively use the commands listed in double quotes e.g."
        ' "command name"'
    )

    # Add commands to the SchemaGenerator object
    for tool in tools:
        schema_generator.add_tool(tool)

    # Add resources to the SchemaGenerator object
    schema_generator.add_resource(
        "Internet access for searches and information gathering."
    )
    schema_generator.add_resource("Long Term memory management.")
    schema_generator.add_resource(
        "GPT-3.5 powered Agents for delegation of simple tasks."
    )
    schema_generator.add_resource("File output.")

    # Add performance evaluations to the SchemaGenerator object
    schema_generator.add_performance_evaluation(
        "Continuously review and analyze your actions "
        "to ensure you are performing to the best of your abilities."
    )
    schema_generator.add_performance_evaluation(
        "Constructively self-criticize your big-picture behavior"
        " constantly."
    )
    schema_generator.add_performance_evaluation(
        "Reflect on past decisions and strategies to refine your"
        " approach."
    )
    schema_generator.add_performance_evaluation(
        "Every command has a cost, so be smart and efficient. "
        "Aim to complete tasks in the least number of steps."
    )

    # Generate the prompt string
    prompt_string = schema_generator.generate_prompt_string()

    return prompt_string

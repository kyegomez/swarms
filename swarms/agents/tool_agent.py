from typing import Any, Optional, Callable
from swarms.tools.json_former import Jsonformer
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="tool_agent")


class ToolAgent:
    """
    Represents a tool agent that performs a specific task using a model and tokenizer.

    Args:
        name (str): The name of the tool agent.
        description (str): A description of the tool agent.
        model (Any): The model used by the tool agent.
        tokenizer (Any): The tokenizer used by the tool agent.
        json_schema (Any): The JSON schema used by the tool agent.
        *args: Variable length arguments.
        **kwargs: Keyword arguments.

    Attributes:
        name (str): The name of the tool agent.
        description (str): A description of the tool agent.
        model (Any): The model used by the tool agent.
        tokenizer (Any): The tokenizer used by the tool agent.
        json_schema (Any): The JSON schema used by the tool agent.

    Methods:
        run: Runs the tool agent for a specific task.

    Raises:
        Exception: If an error occurs while running the tool agent.


    Example:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from swarms import ToolAgent


        model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b")
        tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")

        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "is_student": {"type": "boolean"},
                "courses": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }

        task = "Generate a person's information based on the following schema:"
        agent = ToolAgent(model=model, tokenizer=tokenizer, json_schema=json_schema)
        generated_data = agent.run(task)

        print(generated_data)
    """

    def __init__(
        self,
        name: str = "Function Calling Agent",
        description: str = "Generates a function based on the input json schema and the task",
        model: Any = None,
        tokenizer: Any = None,
        json_schema: Any = None,
        max_number_tokens: int = 500,
        parsing_function: Optional[Callable] = None,
        llm: Any = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            agent_name=name,
            agent_description=description,
            llm=llm,
            **kwargs,
        )
        self.name = name
        self.description = description
        self.model = model
        self.tokenizer = tokenizer
        self.json_schema = json_schema
        self.max_number_tokens = max_number_tokens
        self.parsing_function = parsing_function

    def run(self, task: str, *args, **kwargs):
        """
        Run the tool agent for the specified task.

        Args:
            task (str): The task to be performed by the tool agent.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The output of the tool agent.

        Raises:
            Exception: If an error occurs during the execution of the tool agent.
        """
        try:
            if self.model:
                logger.info(f"Running {self.name} for task: {task}")
                self.toolagent = Jsonformer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    json_schema=self.json_schema,
                    llm=self.llm,
                    prompt=task,
                    max_number_tokens=self.max_number_tokens,
                    *args,
                    **kwargs,
                )

                if self.parsing_function:
                    out = self.parsing_function(self.toolagent())
                else:
                    out = self.toolagent()

                return out
            elif self.llm:
                logger.info(f"Running {self.name} for task: {task}")
                self.toolagent = Jsonformer(
                    json_schema=self.json_schema,
                    llm=self.llm,
                    prompt=task,
                    max_number_tokens=self.max_number_tokens,
                    *args,
                    **kwargs,
                )

                if self.parsing_function:
                    out = self.parsing_function(self.toolagent())
                else:
                    out = self.toolagent()

                return out

            else:
                raise Exception(
                    "Either model or llm should be provided to the"
                    " ToolAgent"
                )

        except Exception as error:
            logger.error(
                f"Error running {self.name} for task: {task}"
            )
            raise error

    def __call__(self, task: str, *args, **kwargs):
        return self.run(task, *args, **kwargs)

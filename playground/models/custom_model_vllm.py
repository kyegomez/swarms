from vllm import LLM
from swarms import AbstractLLM, Agent, ChromaDB


# Making an instance of the VLLM class
class vLLMLM(AbstractLLM):
    """
    This class represents a variant of the Language Model (LLM) called vLLMLM.
    It extends the AbstractLLM class and provides additional functionality.

    Args:
        model_name (str): The name of the LLM model to use. Defaults to "acebook/opt-13b".
        tensor_parallel_size (int): The size of the tensor parallelism. Defaults to 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        model_name (str): The name of the LLM model.
        tensor_parallel_size (int): The size of the tensor parallelism.
        llm (LLM): An instance of the LLM class.

    Methods:
        run(task: str, *args, **kwargs): Runs the LLM model to generate output for the given task.

    """

    def __init__(
        self,
        model_name: str = "acebook/opt-13b",
        tensor_parallel_size: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size

        self.llm = LLM(
            model_name=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
        )

    def run(self, task: str, *args, **kwargs):
        """
        Runs the LLM model to generate output for the given task.

        Args:
            task (str): The task for which to generate output.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The generated output for the given task.

        """
        return self.llm.generate(task)


# Initializing the agent with the vLLMLM instance and other parameters
model = vLLMLM(
    "facebook/opt-13b",
    tensor_parallel_size=4,
)

# Defining the task
task = "What are the symptoms of COVID-19?"

# Running the agent with the specified task
out = model.run(task)


# Integrate Agent
agent = Agent(
    agent_name="Doctor agent",
    agent_description=(
        "This agent provides information about COVID-19 symptoms."
    ),
    llm=model,
    max_loops="auto",
    autosave=True,
    verbose=True,
    long_term_memory=ChromaDB(
        metric="cosine",
        n_results=3,
        output_dir="results",
        docs_folder="docs",
    ),
    stopping_condition="finish",
)

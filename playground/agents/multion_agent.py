import multion

from swarms.models.base_llm import AbstractLLM
from swarms.structs.agent import Agent
from swarms.structs.concurrent_workflow import ConcurrentWorkflow
from swarms.structs.task import Task


class MultiOnAgent(AbstractLLM):
    """
    Represents a multi-on agent that performs browsing tasks.

    Args:
        max_steps (int): The maximum number of steps to perform during browsing.
        starting_url (str): The starting URL for browsing.

    Attributes:
        max_steps (int): The maximum number of steps to perform during browsing.
        starting_url (str): The starting URL for browsing.
    """

    def __init__(
        self,
        multion_api_key: str,
        max_steps: int = 4,
        starting_url: str = "https://www.google.com",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.multion_api_key = multion_api_key
        self.max_steps = max_steps
        self.starting_url = starting_url

        multion.login(
            use_api=True,
            # multion_api_key=self.multion_api_key
            *args,
            **kwargs,
        )

    def run(self, task: str, *args, **kwargs):
        """
        Runs a browsing task.

        Args:
            task (str): The task to perform during browsing.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The response from the browsing task.
        """
        response = multion.browse(
            {
                "cmd": task,
                "url": self.starting_url,
                "maxSteps": self.max_steps,
            },
            *args,
            **kwargs,
        )

        return response.result, response.status, response.lastUrl


# model
model = MultiOnAgent(multion_api_key="")

# out = model.run("search for a recipe")
agent = Agent(
    agent_name="MultiOnAgent",
    description="A multi-on agent that performs browsing tasks.",
    llm=model,
    max_loops=1,
    system_prompt=None,
)


# Task
task = Task(
    agent=agent,
    description=(
        "send an email to vyom on superhuman for a partnership with"
        " multion"
    ),
)

# Swarm
workflow = ConcurrentWorkflow(
    max_workers=1000,
    autosave=True,
    print_results=True,
    return_results=True,
)

# Add task to workflow
workflow.add(task)

# Run workflow
workflow.run()

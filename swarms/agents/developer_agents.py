from swarms.prompts.documentation import DOCUMENTATION_WRITER_SOP
from swarms.prompts.tests import TEST_WRITER_SOP_PROMPT
from swarms.structs.agent import Agent


class UnitTesterAgent:
    """
    This class represents a unit testing agent responsible for generating unit tests for the swarms package.

    Attributes:
    - llm: The low-level model used by the agent.
    - agent_name (str): The name of the agent.
    - agent_description (str): The description of the agent.
    - max_loops (int): The maximum number of loops the agent can run.
    - SOP_PROMPT: The system output prompt used by the agent.
    - agent: The underlying agent object used for running tasks.

    Methods:
    - run(task: str, *args, **kwargs) -> str: Run the agent with the given task and return the response.
    """

    def __init__(
        self,
        llm,
        agent_name: str = "Unit Testing Agent",
        agent_description: str = "This agent is responsible for generating unit tests for the swarms package.",
        max_loops: int = 1,
        sop: str = None,
        module: str = None,
        path: str = None,
        autosave: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.llm = llm
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.max_loops = max_loops
        self.sop = sop
        self.module = module
        self.path = path
        self.autosave = autosave

        self.agent = Agent(
            llm=llm,
            agent_name=agent_name,
            agent_description=agent_description,
            autosave=self.autosave,
            system_prompt=agent_description,
            max_loops=max_loops,
            *args,
            **kwargs,
        )

    def run(self, task: str, module: str, path: str, *args, **kwargs):
        """
        Run the agent with the given task.

        Args:
        - task (str): The task to run the agent with.

        Returns:
        - str: The response from the agent.
        """
        return self.agent.run(
            TEST_WRITER_SOP_PROMPT(task, self.module, self.path),
            *args,
            **kwargs,
        )


class DocumentorAgent:
    """
    This class represents a documentor agent responsible for generating unit tests for the swarms package.

    Attributes:
    - llm: The low-level model used by the agent.
    - agent_name (str): The name of the agent.
    - agent_description (str): The description of the agent.
    - max_loops (int): The maximum number of loops the agent can run.
    - SOP_PROMPT: The system output prompt used by the agent.
    - agent: The underlying agent object used for running tasks.

    Methods:
    - run(task: str, *args, **kwargs) -> str: Run the agent with the given task and return the response.
    """

    def __init__(
        self,
        llm,
        agent_name: str = "Documentor Agent",
        agent_description: str = "This agent is responsible for generating unit tests for the swarms package.",
        max_loops: int = 1,
        sop: str = None,
        module: str = None,
        path: str = None,
        autosave: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.max_loops = max_loops
        self.sop = sop
        self.module = module
        self.path = path
        self.autosave = autosave

        self.agent = Agent(
            llm=llm,
            agent_name=agent_name,
            agent_description=agent_description,
            autosave=self.autosave,
            system_prompt=agent_description,
            max_loops=max_loops,
            *args,
            **kwargs,
        )

    def run(self, task: str, module: str, path: str, *args, **kwargs):
        """
        Run the agent with the given task.

        Args:
        - task (str): The task to run the agent with.

        Returns:
        - str: The response from the agent.
        """
        return self.agent.run(
            DOCUMENTATION_WRITER_SOP(task, self.module) * args,
            **kwargs,
        )

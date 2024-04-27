from typing import List

from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.utils.loguru_logger import logger
from swarms.models.popular_llms import Anthropic, OpenAIChat
from swarms.models.base_llm import BaseLLM
from swarms.memory.base_vectordb import BaseVectorDatabase

boss_sys_prompt = (
    "You're the Swarm Orchestrator, like a project manager of a"
    " bustling hive. When a task arises, you tap into your network of"
    " worker agents who are ready to jump into action. Whether it's"
    " organizing data, handling logistics, or crunching numbers, you"
    " delegate tasks strategically to maximize efficiency. Picture"
    " yourself as the conductor of a well-oiled machine,"
    " orchestrating the workflow seamlessly to achieve optimal"
    " results with your team of dedicated worker agents."
)


class AgentSchema(BaseModel):
    name: str = Field(
        ...,
        title="Name of the agent",
        description="Name of the agent",
    )
    system_prompt: str = (
        Field(
            ...,
            title="System prompt for the agent",
            description="System prompt for the agent",
        ),
    )
    rules: str = Field(
        ...,
        title="Rules",
        description="Rules for the agent",
    )
    llm: str = Field(
        ...,
        title="Language model",
        description="Language model for the agent: `GPT4` or `Claude",
    )

    # tools: List[ToolSchema] = Field(
    #     ...,
    #     title="Tools available to the agent",
    #     description="Either `browser` or `terminal`",
    # )
    # task: str = Field(
    #     ...,
    #     title="Task assigned to the agent",
    #     description="Task assigned to the agent",
    # )
    # TODO: Add more fields here such as the agent's language model, tools, etc.


class HassSchema(BaseModel):
    plan: str = Field(
        ...,
        title="Plan to solve the input problem",
        description="List of steps to solve the problem",
    )
    agents: List[AgentSchema] = Field(
        ...,
        title="List of agents to use for the problem",
        description="List of agents to use for the problem",
    )
    # Rules for the agents
    rules: str = Field(
        ...,
        title="Rules for the agents",
        description="Rules for the agents",
    )


class HiearchicalSwarm(BaseSwarm):
    def __init__(
        self,
        director: Agent = None,
        subordinates: List[Agent] = [],
        workers: List[Agent] = [],
        director_sys_prompt: str = boss_sys_prompt,
        director_name: str = "Swarm Orchestrator",
        director_agent_creation_schema: BaseModel = HassSchema,
        director_llm: BaseLLM = Anthropic,
        communication_protocol: BaseVectorDatabase = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.director = director
        self.subordinates = subordinates
        self.workers = workers
        self.director_sys_prompt = director_sys_prompt
        self.director_name = director_name
        self.director_agent_creation_schema = (
            director_agent_creation_schema
        )
        self.director_llm = director_llm
        self.communication_protocol = communication_protocol

    def create_director(self, *args, **kwargs):
        """
        Create the director agent based on the provided schema.
        """
        name = self.director_name
        system_prompt = self.director_sys_prompt
        director_llm = self.director_llm

        if director_llm == Anthropic:
            Anthropic(*args, **kwargs)
        elif director_llm == OpenAIChat:
            OpenAIChat(*args, **kwargs)

        logger.info(
            f"Creating Director Agent: {name} with system prompt:"
            f" {system_prompt}"
        )

        director = Agent(
            agent_name=name,
            system_prompt=system_prompt,
            llm=director_llm,
            max_loops=1,
            autosave=True,
            dashboard=False,
            verbose=True,
            stopping_token="<DONE>",
        )

        return director

    def create_worker_agents(
        agents: List[AgentSchema],
    ) -> List[Agent]:
        """
        Create and initialize agents based on the provided AgentSchema objects.

        Args:
            agents (List[AgentSchema]): A list of AgentSchema objects containing agent information.

        Returns:
            List[Agent]: The initialized Agent objects.

        """
        agent_list = []
        for agent in agents:
            name = agent.name
            system_prompt = agent.system_prompt

            logger.info(
                f"Creating agent: {name} with system prompt:"
                f" {system_prompt}"
            )

            out = Agent(
                agent_name=name,
                system_prompt=system_prompt,
                # llm=Anthropic(
                #     anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
                # ),
                max_loops=1,
                autosave=True,
                dashboard=False,
                verbose=True,
                stopping_token="<DONE>",
            )

            # network.add_agent(out)
            agent_list.append(out)

        return agent_list

import asyncio
import time
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from swarms.structs.agent import Agent

time_stamp = time.strftime("%Y-%m-%d %H:%M:%S")


class MixtureOfAgentsInput(BaseModel):
    name: str = "MixtureOfAgents"
    description: str = (
        "A class to run a mixture of agents and aggregate their responses."
    )
    reference_agents: List[Dict[str, Any]] = []
    aggregator_agent: Dict[str, Any] = Field(
        ...,
        description="An aggregator agent to be used in the mixture.",
    )
    aggregator_system_prompt: str = ""
    layers: int = 3
    task: str = Field(
        ...,
        description="The task to be processed by the mixture of agents.",
    )
    time_created: str = Field(
        time_stamp,
        description="The time the mixture of agents was created.",
    )


class MixtureOfAgentsOutput(BaseModel):
    id: str = Field(
        ..., description="The ID of the mixture of agents."
    )
    InputConfig: MixtureOfAgentsInput
    output: List[Dict[str, Any]] = []
    time_completed: str = Field(
        time_stamp,
        description="The time the mixture of agents was completed.",
    )


class MixtureOfAgents:
    def __init__(
        self,
        name: str = "MixtureOfAgents",
        description: str = "A class to run a mixture of agents and aggregate their responses.",
        reference_agents: List[Agent] = [],
        aggregator_agent: Agent = None,
        aggregator_system_prompt: str = "",
        layers: int = 3,
    ) -> None:
        """Initialize the Mixture of Agents class with agents and configuration."""
        self.name = name
        self.description = description
        self.reference_agents: List[Agent] = reference_agents
        self.aggregator_agent: Agent = aggregator_agent
        self.aggregator_system_prompt: str = aggregator_system_prompt
        self.layers: int = layers

        self.input_schema = MixtureOfAgentsInput(
            name=name,
            description=description,
            reference_agents=[
                agent.agent_output.model_dump()
                for agent in self.reference_agents
            ],
            aggregator_agent=self.aggregator_agent.agent_output.model_dump(),
            aggregator_system_prompt=self.aggregator_system_prompt,
            layers=self.layers,
            task="",
            time_created=time_stamp,
        )

        self.output_schema = MixtureOfAgentsOutput(
            id="MixtureOfAgents",
            InputConfig=self.input_schema.model_dump(),
            output=[],
        )

    def reliability_check(self) -> None:
        logger.info(
            "Checking the reliability of the Mixture of Agents class."
        )

        if not self.reference_agents:
            raise ValueError("No reference agents provided.")

        if not self.aggregator_agent:
            raise ValueError("No aggregator agent provided.")

        if not self.aggregator_system_prompt:
            raise ValueError("No aggregator system prompt provided.")

        if not self.layers:
            raise ValueError("No layers provided.")

        if self.layers < 1:
            raise ValueError("Layers must be greater than 0.")

        logger.info("Reliability check passed.")
        logger.info("Mixture of Agents class is ready for use.")

    def _get_final_system_prompt(
        self, system_prompt: str, results: List[str]
    ) -> str:
        """Construct a system prompt for layers 2+ that includes the previous responses to synthesize."""
        return (
            system_prompt
            + "\n"
            + "\n".join(
                [
                    f"{i+1}. {str(element)}"
                    for i, element in enumerate(results)
                ]
            )
        )

    async def _run_agent_async(
        self,
        agent: Agent,
        task: str,
        prev_responses: Optional[List[str]] = None,
    ) -> str:
        """Asynchronous method to run a single agent."""
        if prev_responses:
            system_prompt_with_responses = (
                self._get_final_system_prompt(
                    self.aggregator_system_prompt, prev_responses
                )
            )
            agent.system_prompt = system_prompt_with_responses

        response = await asyncio.to_thread(agent.run, task)
        self.output_schema.output.append(
            agent.agent_output.model_dump()
        )

        # Print the agent response
        print(f"Agent {agent.agent_name} response: {response}")
        return response

    async def _run_async(self, task: str) -> None:
        """Asynchronous method to run the Mixture of Agents process."""
        self.input_schema.task = task
        # Initial responses from reference agents
        results: List[str] = await asyncio.gather(
            *[
                self._run_agent_async(agent, task)
                for agent in self.reference_agents
            ]
        )

        # Additional layers of processing
        for _ in range(1, self.layers - 1):
            results = await asyncio.gather(
                *[
                    self._run_agent_async(
                        agent, task, prev_responses=results
                    )
                    for agent in self.reference_agents
                ]
            )

        # Final aggregation using the aggregator agent
        final_result = await self._run_agent_async(
            self.aggregator_agent, task, prev_responses=results
        )
        print(f"Final Aggregated Response: {final_result}")

    def run(self, task: str) -> None:
        """Synchronous wrapper to run the async process."""
        asyncio.run(self._run_async(task))

        return self.output_schema.model_dump_json(indent=4)


# # Example usage:
# if __name__ == "__main__":
#     api_key = os.getenv("OPENAI_API_KEY")

#     # Create individual agents with the OpenAIChat model
#     model1 = OpenAIChat(openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1)
#     model2 = OpenAIChat(openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1)
#     model3 = OpenAIChat(openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1)

#     agent1 = Agent(
#         agent_name="Agent1",
#         system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
#         llm=model1,
#         max_loops=1,
#         autosave=True,
#         dashboard=False,
#         verbose=True,
#         dynamic_temperature_enabled=True,
#         saved_state_path="agent1_state.json",
#         user_name="swarms_corp",
#         retry_attempts=1,
#         context_length=200000,
#         return_step_meta=False
#     )

#     agent2 = Agent(
#         agent_name="Agent2",
#         system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
#         llm=model2,
#         max_loops=1,
#         autosave=True,
#         dashboard=False,
#         verbose=True,
#         dynamic_temperature_enabled=True,
#         saved_state_path="agent2_state.json",
#         user_name="swarms_corp",
#         retry_attempts=1,
#         context_length=200000,
#         return_step_meta=False
#     )

#     agent3 = Agent(
#         agent_name="Agent3",
#         system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
#         llm=model3,
#         max_loops=1,
#         autosave=True,
#         dashboard=False,
#         verbose=True,
#         dynamic_temperature_enabled=True,
#         saved_state_path="agent3_state.json",
#         user_name="swarms_corp",
#         retry_attempts=1,
#         context_length=200000,
#         return_step_meta=False
#     )

#     aggregator_agent = Agent(
#         agent_name="AggregatorAgent",
#         system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
#         llm=model1,
#         max_loops=1,
#         autosave=True,
#         dashboard=False,
#         verbose=True,
#         dynamic_temperature_enabled=True,
#         saved_state_path="aggregator_agent_state.json",
#         user_name="swarms_corp",
#         retry_attempts=1,
#         context_length=200000,
#         return_step_meta=False
#     )

#     # Create the Mixture of Agents class
#     moa = MixtureOfAgents(
#         reference_agents=[agent1, agent2, agent3],
#         aggregator_agent=aggregator_agent,
#         aggregator_system_prompt="""You have been provided with a set of responses from various agents.
#         Your task is to synthesize these responses into a single, high-quality response.""",
#         layers=3
#     )

#     out = moa.run("How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria?")
#     print(out)

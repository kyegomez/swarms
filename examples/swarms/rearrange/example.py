import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from swarms.models import Anthropic
from swarms.structs import Agent
from swarms.structs.rearrange import AgentRearrange

llm = Anthropic(
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"), streaming=True
)


async def sequential():

    agent1 = Agent(
        agent_name="Blog generator",
        system_prompt="Generate a blog post like Stephen King",
        llm=llm,
        dashboard=False,
        streaming_on=True,
    )

    agent2 = Agent(
        agent_name="Summarizer",
        system_prompt="Summarize the blog post",
        llm=llm,
        dashboard=False,
        streaming_on=True,
    )

    flow = f"{agent1.name} -> {agent2.name}"

    agent_rearrange = AgentRearrange(
        [agent1, agent2], flow, verbose=False, logging=False
    )

    result = await agent_rearrange.astream(
        "Generate a short blog post about Muhammad Ali."
    )

    # LEAVING THIS CALL BELOW FOR COMPARISON with "completion-style" .run() approach ;)
    # await agent_rearrange.run(
    #   "Generate a short blog post about Muhammad Ali."
    # )


async def parallel():

    writer1 = Agent(
        agent_name="Writer 1",
        system_prompt="Generate a blog post in the style of J.K. Rowling",
        llm=llm,
        dashboard=False,
    )

    writer2 = Agent(
        agent_name="Writer 2",
        system_prompt="Generate a blog post in the style of Stephen King",
        llm=llm,
        dashboard=False,
    )

    reviewer = Agent(
        agent_name="Reviewer",
        system_prompt="Select the writer that wrote the best story. There can only be one best story.",
        llm=llm,
        dashboard=False,
    )

    flow = f"{writer1.name}, {writer2.name} -> {reviewer.name}"

    agent_rearrange = AgentRearrange(
        [writer1, writer2, reviewer], flow, verbose=False, logging=False
    )

    result = await agent_rearrange.astream(
        "Generate a 1 sentence story about Michael Jordan."
    )

    # LEAVING THIS CALL BELOW FOR COMPARISON with "completion-style" .run() approach ;)
    # result = agent_rearrange.run(
    #   "Generate a short blog post about Michael Jordan."
    # )


# asyncio.run(sequential())
asyncio.run(parallel())

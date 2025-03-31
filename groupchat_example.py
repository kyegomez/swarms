from dotenv import load_dotenv
import os

from swarms.structs.agent import Agent
from swarms.structs.groupchat import GroupChat
from swarms.prompts.multi_agent_collab_prompt import (
    MULTI_AGENT_COLLAB_PROMPT_TWO,
)

if __name__ == "__main__":

    load_dotenv()

    # Get the OpenAI API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    # Example agents
    agent1 = Agent(
        agent_name="Financial-Analysis-Agent",
        description="You are a financial analyst specializing in investment strategies.",
        model_name="gpt-4o-mini",
        max_loops=1,
        autosave=False,
        dashboard=False,
        verbose=True,
        dynamic_temperature_enabled=True,
        user_name="swarms_corp",
        retry_attempts=1,
        context_length=200000,
        output_type="string",
        streaming_on=False,
        max_tokens=15000,
    )

    agent2 = Agent(
        agent_name="Tax-Adviser-Agent",
        description="You are a tax adviser who provides clear and concise guidance on tax-related queries.",
        model_name="gpt-4o-mini",
        max_loops=1,
        autosave=False,
        dashboard=False,
        verbose=True,
        dynamic_temperature_enabled=True,
        user_name="swarms_corp",
        retry_attempts=1,
        context_length=200000,
        output_type="string",
        streaming_on=False,
        max_tokens=15000,
    )

    agents = [agent1, agent2]

    chat = GroupChat(
        name="Investment Advisory",
        description="Financial and tax analysis group",
        agents=agents,
        max_loops=1,
        output_type="all",
    )

    history = chat.run(
        "What are the best Japanese business methodologies to take over a market say like minerals and mining?. I need a 4,000 word report. Work together to write the report."
    )
    # print(history)

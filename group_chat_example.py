
import os
from dotenv import load_dotenv
from swarm_models import OpenAIChat
from swarms import Agent, GroupChat, expertise_based


if __name__ == "__main__":

    load_dotenv()

    # Get the OpenAI API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    # Create an instance of the OpenAIChat class
    model = OpenAIChat(
        openai_api_key=api_key,
        model_name="gpt-4o-mini",
        temperature=0.1,
    )

    # Example agents
    agent1 = Agent(
        agent_name="Financial-Analysis-Agent",
        system_prompt="You are a financial analyst specializing in investment strategies.",
        llm=model,
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
    )

    agent2 = Agent(
        agent_name="Tax-Adviser-Agent",
        system_prompt="You are a tax adviser who provides clear and concise guidance on tax-related queries.",
        llm=model,
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
    )

    agents = [agent1, agent2]

    chat = GroupChat(
        name="Investment Advisory",
        description="Financial and tax analysis group",
        agents=agents,
        speaker_fn=expertise_based,
    )

    history = chat.run(
        "How to optimize tax strategy for investments?"
    )
    print(history.model_dump_json(indent=2))

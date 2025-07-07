import os
from dotenv import load_dotenv
from swarm_models import OpenAIChat
from swarms import Agent, GroupChat


if __name__ == "__main__":

    load_dotenv()

    # Get the OpenAI API key from the environment variable
    api_key = os.getenv("GROQ_API_KEY")

    # Model
    model = OpenAIChat(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=api_key,
        model_name="llama-3.1-70b-versatile",
        temperature=0.1,
    )

    # Example agents
    agent1 = Agent(
        agent_name="Financial-Analysis-Agent",
        system_prompt="You are a friendly financial analyst specializing in investment strategies. Be approachable and conversational.",
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        user_name="swarms_corp",
        output_type="string",
        streaming_on=True,
    )

    agent2 = Agent(
        agent_name="Tax-Adviser-Agent",
        system_prompt="You are a tax adviser who provides clear, concise, and approachable guidance on tax-related queries.",
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        user_name="swarms_corp",
        output_type="string",
        streaming_on=True,
    )

    # agent3 = Agent(
    #     agent_name="Stock-Buying-Agent",
    #     system_prompt="You are a stock market expert who provides insights on buying and selling stocks. Be informative and concise.",
    #     llm=model,
    #     max_loops=1,
    #     dynamic_temperature_enabled=True,
    #     user_name="swarms_corp",
    #     retry_attempts=1,
    #     context_length=200000,
    #     output_type="string",
    #     streaming_on=True,
    # )

    agents = [agent1, agent2]

    chat = GroupChat(
        name="Investment Advisory",
        description="Financial, tax, and stock analysis group",
        agents=agents,
    )

    history = chat.run(
        "How to save on taxes for stocks, ETFs, and mutual funds?"
    )
    print(history.model_dump_json(indent=2))

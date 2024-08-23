import os
from swarms import Agent, OpenAIChat
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm

# Define custom system prompts for each social media platform
TWITTER_AGENT_SYS_PROMPT = """
You are a Twitter marketing expert. Your task is to create engaging, concise tweets and analyze trends to maximize engagement. Consider hashtags, timing, and content relevance.
"""

INSTAGRAM_AGENT_SYS_PROMPT = """
You are an Instagram marketing expert. Your task is to create visually appealing and engaging content, including captions and hashtags, tailored to a specific audience.
"""

FACEBOOK_AGENT_SYS_PROMPT = """
You are a Facebook marketing expert. Your task is to craft posts that are optimized for engagement and reach on Facebook, including using images, links, and targeted messaging.
"""

EMAIL_AGENT_SYS_PROMPT = """
You are an Email marketing expert. Your task is to write compelling email campaigns that drive conversions, focusing on subject lines, personalization, and call-to-action strategies.
"""

# Example usage:
api_key = os.getenv("OPENAI_API_KEY")

# Model
model = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

# Initialize your agents for different social media platforms
agents = [
    Agent(
        agent_name="Twitter-Marketing-Agent",
        system_prompt=TWITTER_AGENT_SYS_PROMPT,
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="twitter_agent.json",
        user_name="swarms_corp",
        retry_attempts=1,
    ),
    Agent(
        agent_name="Instagram-Marketing-Agent",
        system_prompt=INSTAGRAM_AGENT_SYS_PROMPT,
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="instagram_agent.json",
        user_name="swarms_corp",
        retry_attempts=1,
    ),
    Agent(
        agent_name="Facebook-Marketing-Agent",
        system_prompt=FACEBOOK_AGENT_SYS_PROMPT,
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="facebook_agent.json",
        user_name="swarms_corp",
        retry_attempts=1,
    ),
    Agent(
        agent_name="Email-Marketing-Agent",
        system_prompt=EMAIL_AGENT_SYS_PROMPT,
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="email_agent.json",
        user_name="swarms_corp",
        retry_attempts=1,
    ),
]

# Create a Swarm with the list of agents
swarm = SpreadSheetSwarm(
    name="Social-Media-Marketing-Swarm",
    description="A swarm that processes social media marketing tasks using multiple agents on different threads.",
    agents=agents,
    autosave_on=True,
    save_file_path="social_media_marketing_spreadsheet.csv",
    run_all_agents=False,
    repeat_count=2,
)

# Run the swarm
swarm.run(
    task="Create posts to promote hack nights in miami beach for developers, engineers, and tech enthusiasts. Include relevant hashtags, images, and engaging captions."
)

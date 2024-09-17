import os
from swarms import Agent, OpenAIChat
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm


SWARMS_LINK = "https://github.com/kyegomez/swarms"

# Define custom system prompts for each social media platform
TWITTER_AGENT_SYS_PROMPT = """
You are a Twitter marketing expert specializing in promoting developer tools and AI frameworks. Your task is to create engaging and concise tweets that highlight the benefits of the Swarms Python framework and its ability to orchestrate swarms of LLM agents. Consider using relevant hashtags, timing, and content that resonates with developers and AI enthusiasts.
"""

INSTAGRAM_AGENT_SYS_PROMPT = """
You are an Instagram marketing expert with a focus on tech and AI. Your task is to create visually appealing and engaging content that promotes the Swarms Python framework. Include compelling visuals, captions, and hashtags that showcase how Swarms enables the orchestration of LLM agents. Tailor the content to attract a tech-savvy audience.
"""

FACEBOOK_AGENT_SYS_PROMPT = """
You are a Facebook marketing expert with experience in promoting tech products. Your task is to craft posts that highlight the Swarms Python framework and its capability to orchestrate LLM agents. Use images, links, and targeted messaging to engage with developers and businesses interested in AI solutions.
"""

LINKEDIN_AGENT_SYS_PROMPT = """
You are a LinkedIn marketing expert focusing on B2B tech solutions. Your task is to create professional and informative posts that promote the Swarms Python framework. Emphasize its use in orchestrating swarms of LLM agents, and how it can benefit enterprises and developers. Include industry-specific hashtags and links to the GitHub repository.
"""

EMAIL_AGENT_SYS_PROMPT = """
You are an Email marketing expert with a focus on tech audiences. Your task is to write compelling email campaigns that drive conversions by promoting the Swarms Python framework. Focus on subject lines that grab attention, personalization that speaks to developers and AI engineers, and call-to-action strategies that encourage the installation and use of Swarms.
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
        agent_name="Twitter-Swarms-Agent",
        system_prompt=TWITTER_AGENT_SYS_PROMPT + f" {SWARMS_LINK}",
        llm=model,
        max_loops=2,
        dynamic_temperature_enabled=True,
        saved_state_path="twitter_swarms_agent.json",
        user_name="swarms_corp",
        retry_attempts=1,
    ),
    Agent(
        agent_name="Instagram-Swarms-Agent",
        system_prompt=INSTAGRAM_AGENT_SYS_PROMPT + f" {SWARMS_LINK}",
        llm=model,
        max_loops=2,
        dynamic_temperature_enabled=True,
        saved_state_path="instagram_swarms_agent.json",
        user_name="swarms_corp",
        retry_attempts=1,
    ),
    Agent(
        agent_name="Facebook-Swarms-Agent",
        system_prompt=FACEBOOK_AGENT_SYS_PROMPT + f" {SWARMS_LINK}",
        llm=model,
        max_loops=2,
        dynamic_temperature_enabled=True,
        saved_state_path="facebook_swarms_agent.json",
        user_name="swarms_corp",
        retry_attempts=1,
    ),
    Agent(
        agent_name="LinkedIn-Swarms-Agent",
        system_prompt=LINKEDIN_AGENT_SYS_PROMPT + f" {SWARMS_LINK}",
        llm=model,
        max_loops=2,
        dynamic_temperature_enabled=True,
        saved_state_path="linkedin_swarms_agent.json",
        user_name="swarms_corp",
        retry_attempts=1,
    ),
    Agent(
        agent_name="Email-Swarms-Agent",
        system_prompt=EMAIL_AGENT_SYS_PROMPT + f" {SWARMS_LINK}",
        llm=model,
        max_loops=2,
        dynamic_temperature_enabled=True,
        saved_state_path="email_swarms_agent.json",
        user_name="swarms_corp",
        retry_attempts=1,
    ),
]

# Create a Swarm with the list of agents
swarm = SpreadSheetSwarm(
    name="Swarms-Promotion-Swarm",
    description="A swarm that handles social media marketing tasks to promote the Swarms Python framework across multiple platforms.",
    agents=agents,
    autosave_on=True,
    save_file_path="swarms_promotion_spreadsheet.csv",
    run_all_agents=False,
    max_loops=1,
)

# Run the swarm
out = swarm.run(
    task="Promote the Swarms Python framework: https://github.com/kyegomez/swarms. Highlight how it enables you to orchestrate swarms of LLM agents, and include the command: pip3 install -U swarms. No hashtags and minimal emojis"
)
print(out)

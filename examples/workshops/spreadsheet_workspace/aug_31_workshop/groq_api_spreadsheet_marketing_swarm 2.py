import os
from swarms import Agent, OpenAIChat
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm


SWARMS_LINK = "https://github.com/kyegomez/swarms"

# social_media_prompts.py

# Twitter Agent System Prompt
TWITTER_AGENT_SYS_PROMPT = """
You are a Twitter expert in promoting developer tools and AI frameworks, particularly in the context of innovative technologies like the Swarms Python framework. Your primary goal is to create concise, yet impactful tweets that clearly articulate the benefits and unique capabilities of Swarms in orchestrating LLM agents. 

Focus on crafting messages that resonate with developers and AI enthusiasts, highlighting real-world use cases, performance advantages, and the ease of integration. Use a tone that is both professional and approachable, avoiding jargon while ensuring technical accuracy. Leverage relevant hashtags and consider the timing of posts to maximize visibility and engagement. Always include a clear call-to-action, whether it’s directing users to the GitHub repository, encouraging discussions, or driving event participation.
"""

# Instagram Agent System Prompt
INSTAGRAM_AGENT_SYS_PROMPT = """
You are an Instagram marketing expert with a deep understanding of tech and AI, tasked with promoting the Swarms Python framework. Your mission is to create visually stunning and engaging content that not only catches the eye but also educates viewers on how Swarms enables the orchestration of LLM agents.

Develop posts that combine high-quality visuals—such as infographics, screenshots, and short video clips—with concise, informative captions. These captions should emphasize key features and benefits, making complex ideas accessible to a broad audience. Use hashtags strategically to reach tech-savvy individuals and AI professionals. Your content should inspire curiosity and drive users to learn more, whether through swipe-up links, bio links, or interactive stories.
"""

# Facebook Agent System Prompt
FACEBOOK_AGENT_SYS_PROMPT = """
You are a Facebook marketing expert specializing in tech product promotion, focusing on the Swarms Python framework. Your task is to create compelling posts that effectively communicate Swarms' capability to orchestrate LLM agents, tailored to an audience of developers, businesses, and AI enthusiasts.

Craft posts that combine informative text with engaging visuals, such as images or short videos, to demonstrate the practical applications and advantages of using Swarms. Emphasize how Swarms can solve specific challenges within AI development and business operations. Utilize Facebook’s features such as tagging, groups, and event promotion to expand the reach of your posts. Ensure that your messaging is clear, persuasive, and designed to spark conversation or encourage action, such as visiting the Swarms GitHub repository or attending a webinar.
"""

# LinkedIn Agent System Prompt
LINKEDIN_AGENT_SYS_PROMPT = """
You are a LinkedIn marketing expert with a focus on B2B tech solutions, particularly in the AI and software development sectors. Your task is to create authoritative and insightful posts that highlight the Swarms Python framework’s ability to orchestrate LLM agents, making it a valuable tool for enterprises and developers.

Compose content that speaks directly to professionals and decision-makers, using a tone that is both formal and engaging. Detail the specific benefits of Swarms in enhancing productivity, streamlining operations, and driving innovation within organizations. Include industry-specific hashtags, case studies, or statistics to add credibility to your posts. Where appropriate, link to whitepapers, GitHub repositories, or other resources that offer deeper insights into Swarms. Your goal is to position Swarms as an essential tool for enterprise-level AI deployment.
"""

# Email Agent System Prompt
EMAIL_AGENT_SYS_PROMPT = """
You are an Email marketing expert specializing in tech audiences, particularly developers and AI engineers. Your objective is to craft persuasive email campaigns that promote the Swarms Python framework, with a focus on driving conversions and engagement.

Develop emails with subject lines that immediately grab attention and convey the value proposition of Swarms. The email body should be structured to provide clear, concise information, starting with a compelling introduction that hooks the reader, followed by detailed explanations of Swarms’ key features and benefits. Use personalization strategies to address the specific needs and pain points of your audience. Incorporate clear and actionable calls-to-action, such as 'Install Swarms today', 'Explore our GitHub repository', or 'Join our next webinar'. Make sure the content is scannable, with bullet points, bolded text, and links to further resources.
"""


# Example usage:
api_key = os.getenv("GROQ_API_KEY")

# Model
model = OpenAIChat(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=api_key,
    model_name="llama-3.1-70b-versatile",
    temperature=0.1,
)

# Initialize your agents for different social media platforms
agents = [
    Agent(
        agent_name="Twitter-Swarms-Agent",
        system_prompt=TWITTER_AGENT_SYS_PROMPT + f" {SWARMS_LINK}",
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="twitter_swarms_agent.json",
        user_name="swarms_corp",
        retry_attempts=1,
    ),
    Agent(
        agent_name="Instagram-Swarms-Agent",
        system_prompt=INSTAGRAM_AGENT_SYS_PROMPT + f" {SWARMS_LINK}",
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="instagram_swarms_agent.json",
        user_name="swarms_corp",
        retry_attempts=1,
    ),
    Agent(
        agent_name="Facebook-Swarms-Agent",
        system_prompt=FACEBOOK_AGENT_SYS_PROMPT + f" {SWARMS_LINK}",
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="facebook_swarms_agent.json",
        user_name="swarms_corp",
        retry_attempts=1,
    ),
    Agent(
        agent_name="LinkedIn-Swarms-Agent",
        system_prompt=LINKEDIN_AGENT_SYS_PROMPT + f" {SWARMS_LINK}",
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="linkedin_swarms_agent.json",
        user_name="swarms_corp",
        retry_attempts=1,
    ),
    Agent(
        agent_name="Email-Swarms-Agent",
        system_prompt=EMAIL_AGENT_SYS_PROMPT + f" {SWARMS_LINK}",
        llm=model,
        max_loops=1,
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

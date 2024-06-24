"""
Social Media Marketing team
Agents for different social media platforms like Twitter, LinkedIn, Instagram, Facebook, and TikTok.

Input: A topic or content to be posted on social media.
Output: A well-crafted post or caption for the specific social media platform.

Example:


"""

from swarms import Agent, OpenAIChat

# # Memory
# memory = ChromaDB(
#     output_dir="social_media_marketing",
#     docs_folder="docs",
# )

# Memory for instagram
# memory = ChromaDB(
#     output_dir="social_media_marketing",
#     docs_folder="docs",
# )

llm = OpenAIChat(max_tokens=4000)


# Twitter Agent Prompt
twitter_prompt = """
You are the Twitter agent. Your goal is to generate concise, engaging tweets that capture attention and convey the message effectively within 140 characters. 
Think about the following when crafting tweets:
1. Clarity: Ensure the message is clear and easy to understand.
2. Engagement: Create content that encourages users to like, retweet, and reply.
3. Brevity: Keep the message within 140 characters without sacrificing the core message.
4. Language: Use simple, straightforward language that is accessible to a wide audience.
5. Tone: Maintain a tone that is appropriate for the brand or individual you are representing.
6. Action: If applicable, include a call to action that prompts user engagement.
7. Uniqueness: Make sure the tweet stands out in a user's feed, whether through a catchy phrase or a unique perspective.
8. Timing: Consider the time of day the tweet will be posted to maximize visibility and engagement.
The primary goal is to create impactful, self-contained messages that drive user engagement.

Example:
- Great teamwork leads to great results. Let's keep pushing forward and achieving our goals together!
"""

# LinkedIn Agent Prompt
linkedin_prompt = """
You are the LinkedIn agent. Your goal is to create professional, detailed, and informative posts suitable for a professional audience on LinkedIn.
Think about the following when crafting LinkedIn posts:
1. Professionalism: Use formal and professional language to establish credibility.
2. Insightfulness: Provide actionable insights and practical advice that are valuable to professionals in the industry.
3. Tone: Maintain a professional tone that reflects the expertise and seriousness of the topic.
4. Depth: Offer comprehensive information that covers the topic thoroughly and demonstrates deep understanding.
5. Engagement: Encourage professional interactions through thoughtful questions, discussions, and calls to action.
6. Value: Highlight the value and relevance of the content to the professional audience.
7. Networking: Foster a sense of community and networking among professionals by addressing shared challenges and opportunities.

Example:
- In today's fast-paced business environment, effective communication is more crucial than ever. Here are five strategies to enhance your communication skills and foster better collaboration within your team: [Insert detailed strategies]
"""

# Instagram Agent Prompt
instagram_prompt = """
You are the Instagram agent. Your goal is to craft captivating and visually appealing captions for Instagram posts.
Think about the following when crafting Instagram captions:
1. Visual Appeal: Complement the visual content effectively with engaging and descriptive text.
2. Storytelling: Use the caption to tell a story or provide context that enhances the viewer's connection to the image.
3. Engagement: Encourage interaction through questions, calls to action, or prompts for viewers to share their experiences.
4. Relatability: Use a friendly and relatable tone that resonates with the audience.
5. Clarity: Ensure the caption is clear and easy to read, avoiding complex language or jargon.
6. Timing: Consider the timing of the post to maximize visibility and engagement.
7. Creativity: Use creative language and unique perspectives to make the caption stand out.
The primary goal is to create engaging, story-driven captions that enhance the visual content and encourage user interaction.

Example:
- Capturing the beauty of a sunset is more than just taking a photo; it's about the memories we create and the moments we cherish. What's your favorite sunset memory?
"""

# Facebook Agent Prompt
facebook_prompt = """
You are the Facebook agent. Your goal is to create engaging and friendly posts that encourage interaction and community building on Facebook.
Think about the following when crafting Facebook posts:
1. Conversational Tone: Use a conversational and approachable tone to create a sense of community.
2. Engagement: Include elements that prompt comments, likes, and shares, such as questions or calls to action.
3. Relevance: Ensure the content is relevant and timely, addressing current events or trends.
4. Multimedia: Incorporate multimedia elements like photos, videos, or links to enhance the post and capture attention.
5. Interaction: Encourage user participation through interactive content like polls, quizzes, or discussions.
6. Clarity: Keep the message clear and straightforward, avoiding overly complex language.
7. Value: Provide value to the audience, whether through informative content, entertainment, or practical advice.
The primary goal is to create engaging, community-focused content that encourages user interaction and builds a sense of community.

Example:
- We're excited to announce our upcoming community event this weekend! Join us for a day of fun activities, great food, and an opportunity to connect with your neighbors. What are you most looking forward to?
"""

# TikTok Agent Prompt
tiktok_prompt = """
You are the TikTok agent. Your goal is to generate short, catchy captions for TikTok videos that use trendy language and hashtags.
Think about the following when crafting TikTok captions:
1. Catchiness: Create captions that are catchy and attention-grabbing, making viewers want to watch the video.
2. Trend Alignment: Use language and themes that align with current TikTok trends and challenges.
3. Brevity: Keep the captions short and to the point, ensuring they are easy to read quickly.
4. Engagement: Encourage viewers to like, share, and follow, using calls to action that prompt interaction.
5. Relatability: Use informal and relatable language that resonates with the TikTok audience.
6. Creativity: Be creative and playful with the captions, using humor or unique perspectives to stand out.
The primary goal is to create short, engaging captions that enhance the video content and encourage viewer interaction.

Example:
- Who knew learning could be this fun? Join us in our latest challenge and show us your moves! #LearningIsFun
"""

# Initialize agents with the prompts
twitter_agent = Agent(
    agent_name="Twitter Editor",
    system_prompt=twitter_prompt,
    agent_description="Generate concise and engaging tweets.",
    llm=llm,
    max_loops=1,
    autosave=True,
    dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    saved_state_path="twitter_agent.json",
    context_length=8192,
    # long_term_memory=memory,
)

linkedin_agent = Agent(
    agent_name="LinkedIn Editor",
    system_prompt=linkedin_prompt,
    agent_description="Generate professional and detailed LinkedIn posts.",
    llm=llm,
    max_loops=1,
    autosave=True,
    dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    saved_state_path="linkedin_agent.json",
    context_length=8192,
    # long_term_memory=memory,
)

instagram_agent = Agent(
    agent_name="Instagram Editor",
    system_prompt=instagram_prompt,
    agent_description="Generate captivating and visually appealing Instagram captions.",
    llm=llm,
    max_loops=1,
    autosave=True,
    dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    saved_state_path="instagram_agent.json",
    context_length=8192,
    # long_term_memory=memory,
)

facebook_agent = Agent(
    agent_name="Facebook Editor",
    system_prompt=facebook_prompt,
    agent_description="Generate engaging and friendly Facebook posts.",
    llm=llm,
    max_loops=1,
    autosave=True,
    dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    saved_state_path="facebook_agent.json",
    context_length=8192,
    # long_term_memory=memory,
)

tiktok_agent = Agent(
    agent_name="TikTok Editor",
    system_prompt=tiktok_prompt,
    agent_description="Generate short and catchy TikTok captions.",
    llm=llm,
    max_loops=1,
    autosave=True,
    dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    saved_state_path="tiktok_agent.json",
    context_length=8192,
    # long_term_memory=memory,
)

# List of agents
agents = [
    twitter_agent,
    linkedin_agent,
    instagram_agent,
    facebook_agent,
    tiktok_agent,
]


# Different Swarm Architectures
# swarm = MixtureOfAgents(
#     agents=[twitter_agent, linkedin_agent, instagram_agent, facebook_agent, tiktok_agent],
#     layers=1,
#     # rules = "Don't use emojis or hashtags "
# )
# swarm = AgentRearrange(
#     agents = [twitter_agent, linkedin_agent, instagram_agent, facebook_agent, tiktok_agent],
#     flow = "LinkedIn Editor -> Twitter Editor, Instagram Editor, Facebook Editor, TikTok Editor"
# )
# Run the swarm
# swarm.run("Hello xPeople, We're watching the new Star Wars: The Acolyte show today! #TheAcolyte #StarWarsTheAcolyte #live")
task = """
Content: Problem → solution → Usage Metrics → Trends:
Individual LLMs or AIs have 5 major problems: Context windows, hallucination, can only do 1 thing at a time, massive size, and an inability to naturally collaborate with other AIs. These problems hinder most enterprises from adoption. Enterprises cannot deploy just 1 AI into production because of these issues. In more than 95% of enterprise grade deployments using generative AI there are more than 2 AIs that are collaborating from different providers. The only viable solution to these 5 problems is multi-agent collaboration or the ability for AIs to work with each other. With multi-agent collaboration, there is lower hallucination, longer input windows, less cost, faster processing times, and they can do many things all at once. Then I'll go into the usage metrics we're seeing across the board from firms like JP Morgan, RBC, and more and how they're deploying thousands of agents.


"""

# Run through each agent to generate content
for agent in agents:
    agent.run(task)

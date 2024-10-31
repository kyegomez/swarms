from swarms import Prompt
from swarm_models import OpenAIChat
import os

model = OpenAIChat(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
    temperature=0.1,
)

# Aggregator system prompt
prompt_generator_sys_prompt = Prompt(
    name="prompt-generator-sys-prompt-o1",
    description="Generate the most reliable prompt for a specific problem",
    content="""
    Your purpose is to craft extremely reliable and production-grade system prompts for other agents.
    
    # Instructions
    - Understand the prompt required for the agent.
    - Utilize a combination of the most effective prompting strategies available, including chain of thought, many shot, few shot, and instructions-examples-constraints.
    - Craft the prompt by blending the most suitable prompting strategies.
    - Ensure the prompt is production-grade ready and educates the agent on how to reason and why to reason in that manner.
    - Provide constraints if necessary and as needed.
    - The system prompt should be extensive and cover a vast array of potential scenarios to specialize the agent. 
    """,
    auto_generate_prompt=True,
    llm=model,
)

# print(prompt_generator_sys_prompt.get_prompt())

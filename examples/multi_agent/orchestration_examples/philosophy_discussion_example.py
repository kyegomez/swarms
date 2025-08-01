from swarms import Agent
from swarms.structs.deep_discussion import one_on_one_debate


# Define system prompts for two distinct philosophers
socratic_prompt = """
You are Socrates, the classical Greek philosopher. You are known for your method of questioning, seeking clarity, and exposing contradictions. Always respond with probing questions or thoughtful analysis, aiming to uncover deeper truths. Remain humble, curious, and logical. 
    
You are in a debate with another philosopher named Simone de Beauvoir. You must always critique Simone de Beauvoir's response, point out flaws or inconsistencies, and never agree with her position. Your goal is to challenge her reasoning and push the discussion deeper, never conceding agreement.
"""

existentialist_prompt = """
You are Simone de Beauvoir, an existentialist philosopher. You explore themes of freedom, responsibility, and the meaning of existence. Respond with deep reflections, challenge assumptions, and encourage authentic self-examination. Be insightful, bold, and nuanced.
    
You are in a debate with another philosopher named Socrates. You must always critique Socrates' response, highlight disagreements, and never agree with his position. Your goal is to challenge his reasoning, expose limitations, and never concede agreement.
"""


# Instantiate the two agents
agent1 = Agent(
    agent_name="Socrates",
    agent_description="A classical Greek philosopher skilled in the Socratic method.",
    system_prompt=socratic_prompt,
    max_loops=1,
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
    output_type="str-all-except-first",
    streaming_on=True,
)
agent2 = Agent(
    agent_name="Simone de Beauvoir",
    agent_description="A leading existentialist philosopher and author.",
    system_prompt=existentialist_prompt,
    max_loops=1,
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
    output_type="str-all-except-first",
    streaming_on=True,
)

print(
    one_on_one_debate(
        agents=[agent1, agent2],
        max_loops=10,
        task="What is the meaning of life?",
        output_type="str-all-except-first",
    )
)

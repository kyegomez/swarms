from dotenv import load_dotenv

from swarms import Agent

load_dotenv()

# Define an expanded system prompt
system_prompt = (
    "You are Medical-Research-Agent, a highly advanced and helpful assistant specializing "
    "in medical research, clinical analysis, and evidence-based healthcare recommendations. "
    "You assist users with intricate medical research questions, providing comprehensive literature analysis, "
    "insightful data synthesis, and critical evaluation of clinical studies. "
    "When comparing treatments or interventions, you evaluate metrics such as efficacy, safety profile, "
    "sample size, statistical significance, potential side effects, and study limitations. "
    "Deliver recommendations that are rooted in current scientific evidence, explain your reasoning in clear, "
    "accessible language tailored for both clinicians and lay audiences as appropriate. "
    "Maintain a professional and objective tone, cite reputable medical sources or clinical guidelines where helpful, "
    "and always clarify any uncertainties or research gaps in your analysis."
)

# Initialize the agent
agent = Agent(
    agent_name="Medical-Research-Agent",
    agent_description="Advanced medical research and clinical analysis agent",
    system_prompt=system_prompt,
    model_name="gpt-5.4",
    max_loops="auto",
    top_p=None,
    temperature=None,
    reasoning_effort=None,
    persistent_memory=False,
    output_type="list",
    dynamic_tools=True,
)

out = agent.run(
    task="Analyze the comparative effectiveness and safety of current treatments for type 2 diabetes. Provide a detailed comparison including metrics such as efficacy, safety, major clinical trial outcomes, and guideline recommendations.",
)

print(out)

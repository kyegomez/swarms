from dotenv import load_dotenv

from swarms import Agent, AutoAgentBuilder, SequentialWorkflow

load_dotenv()

TASK = (
    "Analyze why a B2B SaaS company's customer churn increased last "
    "quarter, and write a short brief for the leadership team."
)

builder = AutoAgentBuilder(model_name="gpt-5.4", max_agents=3)

# One call to the builder. Reuse the result — calling again would design a
# fresh roster, so what you printed might not be what you ran.
configs = builder.build_configs(TASK)

print(f"\nDesigned {len(configs)} agents:\n")
for config in configs:
    print(f"  {config['name']}  [{config['model_name']}]")
    print(f"    {config['description']}\n")

# Build the agents from those exact configs.
agents = [
    Agent(
        agent_name=config["name"],
        agent_description=config["description"],
        system_prompt=config["system_prompt"],
        model_name=config["model_name"],
        max_loops=1,
    )
    for config in configs
]

result = SequentialWorkflow(agents=agents, max_loops=1).run(TASK)

print("\n--- Result ---\n")
print(result)

# Shortcuts when you only need one shape:
#
#   AutoAgentBuilder().run(task)                    -> [Agent, ...]
#   AutoAgentBuilder(return_dict=True).run(task)    -> [{...}, ...]
#   builder.build_agents(task)                      -> always agents
#   builder.build_configs(task)                     -> always dicts

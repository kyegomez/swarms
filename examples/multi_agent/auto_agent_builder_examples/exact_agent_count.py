from dotenv import load_dotenv

from swarms import AutoAgentBuilder

load_dotenv()

TASK = "Evaluate whether a mid-size logistics company is worth acquiring."


# A ceiling. The builder decides how many it actually needs.
ceiling = AutoAgentBuilder(max_agents=5, return_dict=True)

# An exact count. The builder must split the work to reach it.
exact = AutoAgentBuilder(num_agents=5, return_dict=True)


for label, builder in [
    ("max_agents=5", ceiling),
    ("num_agents=5", exact),
]:
    configs = builder.run(TASK)
    names = ", ".join(c["name"] for c in configs)
    print(f"{label:16} -> {len(configs)} agents: {names}")

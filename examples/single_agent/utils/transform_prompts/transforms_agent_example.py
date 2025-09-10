from swarms import Agent
from swarms.structs.transforms import TransformConfig

# Initialize the agent with message transforms enabled
# This will automatically handle context size limits using middle-out compression
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    model_name="claude-sonnet-4-20250514",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    streaming_on=False,
    print_on=False,
    # Enable message transforms for handling context limits
    transforms=TransformConfig(
        enabled=True,
        method="middle-out",
        model_name="claude-sonnet-4-20250514",
        preserve_system_messages=True,
        preserve_recent_messages=2,
    ),
)

# Alternative way to configure transforms using dictionary
# agent_with_dict_transforms = Agent(
#     agent_name="Trading-Agent-Dict",
#     model_name="gpt-4o",
#     max_loops=1,
#     transforms={
#         "enabled": True,
#         "method": "middle-out",
#         "model_name": "gpt-4o",
#         "preserve_system_messages": True,
#         "preserve_recent_messages": 3,
#     },
# )

out = agent.run(
    task="What are the top five best energy stocks across nuclear, solar, gas, and other energy sources?",
)

print(out)

# The transforms feature provides:
# 1. Automatic context size management for models with token limits
# 2. Message count management for models like Claude with 1000 message limits
# 3. Middle-out compression that preserves important context (beginning and recent messages)
# 4. Smart model selection based on context requirements
# 5. Detailed logging of compression statistics

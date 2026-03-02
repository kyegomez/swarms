from swarms.agents.reasoning_duo import ReasoningDuo
from loguru import logger
import time
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Configure loguru to display more detailed information
logger.remove()
logger.add(lambda msg: print(msg), level="DEBUG")

print("===== agent pool mechanism demonstration (Claude model) =====")

# Example 1: When creating a agent for the first time, a new agent instance will be created and added to the pool
print("\n[example1]: Creating the first ReasoningDuo instance (Configuration A)")
duo_a1 = ReasoningDuo(
    system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
    model_names=["claude-3-5-haiku-20241022", "claude-3-7-sonnet-20250219"],
    reasoning_model_name="claude-3-5-haiku-20241022",
    agent_name="finance-agent"
)

# use first agent to run a task
print("\n[Example 1] Run a task using the first agent:")
result = duo_a1.run(
    "What is the best possible financial strategy to maximize returns but minimize risk? Give a list of etfs to invest in and the percentage of the portfolio to allocate to each etf."
)

# example2: Create a second instance of the same configuration while the first instance is still in use
print("\n[example2]: Creating the second ReasoningDuo instance (Configuration A)")
duo_a2 = ReasoningDuo(
    system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
    model_names=["claude-3-5-haiku-20241022", "claude-3-7-sonnet-20250219"],
    reasoning_model_name="claude-3-5-haiku-20241022",
    agent_name="finance-agent"
)

# example3: Create a third instance with a different configuration, which will create a new agent
print("\n[example3]: Creating a ReasoningDuo instance with a different configuration:")
duo_b = ReasoningDuo(
    system_prompt="You are an expert financial advisor helping with investment strategies.",
    model_names=["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"],  # 不同的模型组合
    reasoning_model_name="claude-3-5-haiku-20241022",
    agent_name="expert-finance-agent"
)

# example4: Disable agent pool reuse
print("\n[example4]: Creating a ReasoningDuo instance with agent reuse disabled:")
duo_c = ReasoningDuo(
    system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
    model_names=["claude-3-5-haiku-20241022", "claude-3-7-sonnet-20250219"],
    reasoning_model_name="claude-3-5-haiku-20241022",
    agent_name="finance-agent",
    reuse_agents=False  # Disable agent reuse
)

# Release the first instance's agent back to the pool
print("\n[example5]: Releasing the first instance's agent back to the pool:")
del duo_a1  # This will trigger the __del__ method and release the agent back to the pool

# Wait for a second to ensure cleanup is complete
time.sleep(1)

# Example 6: Create a new instance with the same configuration as the first one, which should reuse the agent from the pool
print("\n[example6]: Creating a new instance with the same configuration as the first one, which should reuse the agent from the pool:")
duo_a3 = ReasoningDuo(
    system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
    model_names=["claude-3-5-haiku-20241022", "claude-3-7-sonnet-20250219"],
    reasoning_model_name="claude-3-5-haiku-20241022",
    agent_name="finance-agent"
)

# use batched_run to execute tasks
print("\n[example7]: Using the reused agent to run batched tasks:")
results = duo_a3.batched_run(
    [
        "What is the best possible financial strategy for a conservative investor nearing retirement?",
        "What is the best possible financial strategy for a young investor with high risk tolerance?"
    ]
)

# Configure pool parameters
print("\n[example8]: Configuring agent pool parameters:")
ReasoningDuo.configure_pool(cleanup_interval=180, max_idle_time=600)

# Display current pool status
print("\n[example9]: Displaying current pool status:")
print(f"Reasoning agent pool size: {len(ReasoningDuo._reasoning_agent_pool)}")
print(f"Main agent pool size: {len(ReasoningDuo._main_agent_pool)}")

# Clear all pools
print("\n[example10]: Clearing all agent pools:")
ReasoningDuo.clear_pools()
print(f"Reasoning agent pool size after clearing: {len(ReasoningDuo._reasoning_agent_pool)}")
print(f"Main agent pool size after clearing: {len(ReasoningDuo._main_agent_pool)}")
import asyncio
import time
import psutil

from swarms.structs.agent import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

# Initialize the agent (no external imports or env lookups needed here)
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
    output_type="string",
    streaming_on=False,
    model_name="gpt-4o-mini",
)


# Helper decorator to measure time and memory usage
def measure_time_and_memory(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        mem_mb = psutil.Process().memory_info().rss / 1024**2
        print(
            f"[{func.__name__}] Time: {elapsed:.2f}s | Memory: {mem_mb:.2f} MB"
        )
        return result

    return wrapper


# Async wrapper using asyncio.to_thread for the blocking call
@measure_time_and_memory
async def run_agent_async():
    return await asyncio.to_thread(
        agent.run,
        "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria?",
    )


# Threaded wrapper simply runs the async version in the event loop again
@measure_time_and_memory
def run_agent_in_thread():
    asyncio.run(run_agent_async())


if __name__ == "__main__":
    # 1) Run asynchronously
    asyncio.run(run_agent_async())
    # 2) Then run again via the threaded wrapper
    run_agent_in_thread()

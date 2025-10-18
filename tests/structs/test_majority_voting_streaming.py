from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms.structs.majority_voting import MajorityVoting
from dotenv import load_dotenv

def streaming_callback(agent_name: str, chunk: str, is_final: bool):
    # Chunk buffer static per call (reset each session)
    if not hasattr(streaming_callback, "_buffer"):
        streaming_callback._buffer = ""
        streaming_callback._buffer_size = 0

    min_chunk_size = 512  # or any large chunk size you want

    if chunk:
        streaming_callback._buffer += chunk
        streaming_callback._buffer_size += len(chunk)
    if streaming_callback._buffer_size >= min_chunk_size or is_final:
        if streaming_callback._buffer:
            print(streaming_callback._buffer, end="", flush=True)
            streaming_callback._buffer = ""
            streaming_callback._buffer_size = 0
    if is_final:
        print()

load_dotenv()


# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
    user_name="swarms_corp",
    retry_attempts=3,
    context_length=8192,
    return_step_meta=False,
    output_type="str",  # "json", "dict", "csv" OR "string" "yaml" and
    auto_generate_prompt=False,  # Auto generate prompt for the agent based on name, description, and system prompt, task
    max_tokens=4000,  # max output tokens
    saved_state_path="agent_00.json",
    interactive=False,
    streaming_on=True,  #if concurrent agents want to be streamed
)

swarm = MajorityVoting(agents=[agent, agent, agent])

swarm.run(
    "Create a table of super high growth opportunities for AI. I have $40k to invest in ETFs, index funds, and more. Please create a table in markdown.",
    streaming_callback=streaming_callback,

)


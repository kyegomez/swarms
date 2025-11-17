from swarms import Agent
from swarms_tools import scrape_and_format_sync
from swarms.structs.concurrent_workflow import ConcurrentWorkflow
import time

agent = Agent(
    agent_name="Web Scraper Agent",
    model_name="groq/moonshotai/kimi-k2-instruct",
    system_prompt="You are a web scraper agent. You are given a URL and you need to scrape the website and return the data in a structured format. The format type should be full",
    tools=[scrape_and_format_sync],
    dynamic_context_window=True,
    dynamic_temperature_enabled=True,
    max_loops=1,
    streaming_on=True,  # Enable streaming mode
    print_on=False,  # Prevent direct printing (let callback handle it)
)


def streaming_callback(agent_name: str, chunk: str, is_final: bool):
    """
    Callback function to handle streaming output from agents.

    Args:
        agent_name (str): Name of the agent producing the output
        chunk (str): Chunk of output text
        is_final (bool): Whether this is the final chunk (completion signal)
    """
    if is_final:
        print(f"\n[{agent_name}] - COMPLETED")
        return

    if chunk:
        # Print timestamp with agent name for each chunk
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        print(
            f"[{timestamp}] [{agent_name}] {chunk}",
            end="",
            flush=True,
        )
    else:
        # Debug: print when chunk is empty but not final
        print(f"[{agent_name}] - EMPTY CHUNK", end="", flush=True)


# Alternative simple callback (uncomment to use instead):
# def simple_streaming_callback(agent_name: str, chunk: str, is_final: bool):
#     """Simple callback that just prints agent output without timestamps."""
#     if is_final:
#         print(f"\n--- {agent_name} FINISHED ---")
#     elif chunk:
#         print(f"[{agent_name}] {chunk}", end="", flush=True)


# For saving to file (uncomment to use):
# import os
# def file_streaming_callback(agent_name: str, chunk: str, is_final: bool):
#     """Callback that saves streaming output to separate files per agent."""
#     if not os.path.exists('agent_outputs'):
#         os.makedirs('agent_outputs')
#
#     filename = f"agent_outputs/{agent_name.replace(' ', '_')}.txt"
#
#     with open(filename, 'a', encoding='utf-8') as f:
#         if is_final:
#             f.write(f"\n--- COMPLETED ---\n")
#         elif chunk:
#             f.write(chunk)


swarm = ConcurrentWorkflow(
    agents=[agent, agent],
    name="Web Scraper Swarm",
    description="This swarm is used to scrape the web and return the data in a structured format.",
)

swarm.run(
    task="Scrape swarms.ai website and provide a full report of the company does. The format type should be full.",
    streaming_callback=streaming_callback,
)

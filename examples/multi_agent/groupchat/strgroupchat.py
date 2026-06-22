from dotenv import load_dotenv
load_dotenv()

from swarms import RESPOND_TOOL, Agent, GroupChat


def streaming_callback(agent_name: str, content: str, is_final: bool):
    if is_final:
        print(f"[{agent_name}] END")
    elif content:
        print(f"[{agent_name}] {content}")


a1 = Agent(
    agent_name="Optimist",
    system_prompt="You argue for the benefits and positive aspects.",
    model_name="gpt-4o-mini",
    max_loops=2,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
    output_type="final",
    streaming_on=True,
)

a2 = Agent(
    agent_name="Pessimist",
    system_prompt="You argue for the risks and downsides.",
    model_name="gpt-4o-mini",
    max_loops=2,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
    output_type="final",
    streaming_on=True,
)

a3 = Agent(
    agent_name="Realist",
    system_prompt="You provide balanced analysis and moderate the discussion.",
    model_name="gpt-4o-mini",
    max_loops=2,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
    output_type="final",
    streaming_on=True,
)

chat = GroupChat(
    agents=[a1, a2, a3],
    max_loops=10,
    threshold=0.5,
    verbose=True,
    streaming_callback=streaming_callback,
    auto_equip=False,
)

result = chat.run("Should we adopt AI for medical diagnosis?")
print("\n--- Final Result ---")
print(result)

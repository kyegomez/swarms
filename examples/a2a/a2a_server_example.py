"""Deploy an agent as an A2A server.

Run this file, then call it from another process with a2a_client_example.py.
The agent card is public at http://127.0.0.1:8000/.well-known/agent-card.json.
"""

from swarms import A2AServer, Agent

researcher = Agent(
    agent_name="Researcher",
    agent_description="Answers research questions in a short paragraph.",
    system_prompt="Answer the question in one short, factual paragraph.",
    model_name="gpt-5.4-mini",
    max_loops=1,
    print_on=False,
)

if __name__ == "__main__":
    server = A2AServer(
        researcher,
        api_keys=["demo-key"],
        host="127.0.0.1",
        port=8000,
        # Everything below is optional; these are the defaults.
        card_path="/.well-known/agent-card.json",
        rpc_path="/a2a",
        agent_version="1.0.0",
        input_modes=["text/plain"],
        output_modes=["text/plain"],
    )
    # Blocks until interrupted with Ctrl+C.
    server.run()

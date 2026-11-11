"""Serve an agent over A2A and call it from the same script."""

import socket

from swarms import A2AClient, A2AServer, Agent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


poet = Agent(
    agent_name="Poet",
    agent_description="Writes a two-line poem about the topic it is given.",
    system_prompt="Write exactly two short lines of poetry about the topic. Nothing else.",
    model_name="gpt-5.4-mini",
    max_loops=1,
    print_on=False,
)

if __name__ == "__main__":
    port = free_port()

    # The with block starts the server in the background and stops it on exit.
    with A2AServer(poet, api_keys=["demo-key"], port=port):
        remote = A2AClient(
            f"http://127.0.0.1:{port}", api_key="demo-key"
        )
        print(f"Calling {remote.agent_name} at {remote.url}")
        print(remote.run("Write a poem about lighthouses."))

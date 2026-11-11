"""Call a remote A2A agent.

Start a2a_server_example.py first, or point the URL at any A2A 1.0 agent.
"""

from swarms import A2AClient

remote = A2AClient(
    "http://127.0.0.1:8000",
    api_key="demo-key",
    # Optional: seconds to wait for a reply (default 300).
    timeout=120.0,
)

print(f"Connected to {remote.agent_name}: {remote.agent_description}")
print(f"JSON-RPC endpoint: {remote.url}")

answer = remote.run("What is the A2A protocol used for?")
print(answer)

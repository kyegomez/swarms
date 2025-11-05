"""
Simple AOP Server with Custom Authentication Callback

The auth_callback function determines ALL authentication logic.
If you provide auth_callback, authentication is enabled.
If you don't provide it, no authentication is required.
"""

from swarms import Agent
from swarms.structs.aop import AOP


# EXAMPLE: This function governs ALL security
def custom_auth(token: str) -> bool:
    """
    Your custom authentication logic goes here.
    Return True to allow access, False to deny.

    This function determines everything:
    - What tokens are valid
    - Token format (API key, JWT, whatever)
    - Any additional validation logic
    """

    valid_tokens = {
        "mytoken123",
        "anothertoken456",
    }
    return token in valid_tokens


# Create agents
agent = Agent(
    agent_name="Research-Agent",
    model_name="claude-sonnet-4-5-20250929",
    max_loops=1,
    system_prompt="You are a helpful research assistant.",
    temperature=0.7,
    top_p=None, 
)


server = AOP(
    server_name="SimpleAuthServer",
    port=5932,
    auth_callback=custom_auth,
)

server.add_agent(agent)

print("\n🚀 Server starting on port 5932")
print("🔐 Authentication: ENABLED")
print("✅ Valid tokens: mytoken123, anothertoken456\n")

server.run()

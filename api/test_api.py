import requests
import json
from time import sleep

BASE_URL = "http://0.0.0.0:8000/v1"


def make_request(method, endpoint, data=None):
    """Helper function to make requests with error handling"""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(
            f"Error making {method} request to {endpoint}: {str(e)}"
        )
        if hasattr(e.response, "text"):
            print(f"Response text: {e.response.text}")
        return None


def create_agent():
    """Create a test agent"""
    data = {
        "agent_name": "test_agent",
        "model_name": "gpt-4",
        "system_prompt": "You are a helpful assistant",
        "description": "Test agent",
        "temperature": 0.7,
        "max_loops": 1,
        "tags": ["test"],
    }
    return make_request("POST", "/v1/agent", data)


def list_agents():
    """List all agents"""
    return make_request("GET", "/v1/agents")


def test_completion(agent_id):
    """Test a completion with the agent"""
    data = {
        "prompt": "Say hello!",
        "agent_id": agent_id,
        "max_tokens": 100,
    }
    return make_request("POST", "/v1/agent/completions", data)


def get_agent_metrics(agent_id):
    """Get metrics for an agent"""
    return make_request("GET", f"/v1/agent/{agent_id}/metrics")


def delete_agent(agent_id):
    """Delete an agent"""
    return make_request("DELETE", f"/v1/agent/{agent_id}")


def run_tests():
    print("Starting API tests...")

    # Create an agent
    print("\n1. Creating agent...")
    agent_response = create_agent()
    if not agent_response:
        print("Failed to create agent")
        return

    agent_id = agent_response.get("agent_id")
    print(f"Created agent with ID: {agent_id}")

    # Give the server a moment to process
    sleep(2)

    # List agents
    print("\n2. Listing agents...")
    agents = list_agents()
    print(f"Found {len(agents)} agents")

    # Test completion
    if agent_id:
        print("\n3. Testing completion...")
        completion = test_completion(agent_id)
        if completion:
            print(
                f"Completion response: {completion.get('response')}"
            )

        print("\n4. Getting agent metrics...")
        metrics = get_agent_metrics(agent_id)
        if metrics:
            print(f"Agent metrics: {json.dumps(metrics, indent=2)}")

        # Clean up
        # print("\n5. Cleaning up - deleting agent...")
        # delete_result = delete_agent(agent_id)
        # if delete_result:
        #     print("Successfully deleted agent")


if __name__ == "__main__":
    run_tests()

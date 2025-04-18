
import unittest
from swarms.structs.agent import Agent

class TestBasicExample(unittest.TestCase):
    def setUp(self):
        self.agent = Agent(
            agent_name="Test-Agent",
            agent_description="A test agent",
            system_prompt="You are a helpful assistant.",
            model_name="gpt-4",
        )
    
    def test_agent_initialization(self):
        self.assertEqual(self.agent.agent_name, "Test-Agent")
        self.assertEqual(self.agent.agent_description, "A test agent")

    def test_agent_run(self):
        response = self.agent.run("What is 2+2?")
        self.assertIsNotNone(response)

if __name__ == "__main__":
    unittest.main()

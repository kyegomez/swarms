import unittest
import swarms
from swarms.agents.workers.worker import WorkerNode
from swarms.agents.boss.boss_agent import BossNode

class TestSwarms(unittest.TestCase):
    def setUp(self):
        self.swarm = swarms.Swarms('fake_api_key')

    def test_initialize_llm(self):
        llm = self.swarm.initialize_llm(swarms.ChatOpenAI)
        self.assertIsNotNone(llm)

    def test_initialize_tools(self):
        tools = self.swarm.initialize_tools(swarms.ChatOpenAI)
        self.assertIsNotNone(tools)

    def test_initialize_vectorstore(self):
        vectorstore = self.swarm.initialize_vectorstore()
        self.assertIsNotNone(vectorstore)

    def test_run_swarms(self):
        objective = "Do a web search for 'OpenAI'"
        result = self.swarm.run_swarms(objective)
        self.assertIsNotNone(result)


class TestWorkerNode(unittest.TestCase):
    def setUp(self):
        swarm = swarms.Swarms('fake_api_key')
        worker_tools = swarm.initialize_tools(swarms.ChatOpenAI)
        vectorstore = swarm.initialize_vectorstore()
        self.worker_node = swarm.initialize_worker_node(worker_tools, vectorstore)

    def test_create_agent(self):
        self.worker_node.create_agent("Worker 1", "Assistant", False, {})
        self.assertIsNotNone(self.worker_node.agent)

    def test_run(self):
        tool_input = {'prompt': "Search the web for 'OpenAI'"}
        result = self.worker_node.run(tool_input)
        self.assertIsNotNone(result)


class TestBossNode(unittest.TestCase):
    def setUp(self):
        swarm = swarms.Swarms('fake_api_key')
        worker_tools = swarm.initialize_tools(swarms.ChatOpenAI)
        vectorstore = swarm.initialize_vectorstore()
        worker_node = swarm.initialize_worker_node(worker_tools, vectorstore)
        self.boss_node = swarm.initialize_boss_node(vectorstore, worker_node)

    def test_create_task(self):
        task = self.boss_node.create_task("Do a web search for 'OpenAI'")
        self.assertIsNotNone(task)

    def test_execute_task(self):
        task = self.boss_node.create_task("Do a web search for 'OpenAI'")
        result = self.boss_node.execute_task(task)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()

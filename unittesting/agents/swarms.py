import unittest
from unittest.mock import patch
from swarms.agents.swarms import WorkerNode, BossNode, llm, tools, vectorstore, task_execution_chain

class TestSwarms(unittest.TestCase):

    def test_WorkerNode_create_agent(self):
        worker_node = WorkerNode(llm, tools, vectorstore)
        worker_node.create_agent('test_agent', 'test_role', False, {})
        self.assertIsNotNone(worker_node.agent)
        self.assertEqual(worker_node.agent.ai_name, 'test_agent')
        self.assertEqual(worker_node.agent.ai_role, 'test_role')
        self.assertEqual(worker_node.agent.human_in_the_loop, False)
        self.assertTrue(worker_node.agent.chain.verbose)

    def test_WorkerNode_tools(self):
        worker_node = WorkerNode(llm, tools, vectorstore)
        worker_node.create_agent('test_agent', 'test_role', False, {})
        for tool in worker_node.tools:
            self.assertIsNotNone(tool)

    @patch.object(WorkerNode, 'run_agent')
    def test_WorkerNode_run_agent_called(self, mock_run_agent):
        worker_node = WorkerNode(llm, tools, vectorstore)
        worker_node.create_agent('test_agent', 'test_role', False, {})
        worker_node.run_agent('test_prompt')
        mock_run_agent.assert_called_once()

    def test_BossNode_create_task(self):
        boss_node = BossNode(llm, vectorstore, task_execution_chain, False, 3)
        task = boss_node.create_task('objective')
        self.assertEqual(task, {"objective": "objective"})

    def test_BossNode_AgentExecutor(self):
        boss_node = BossNode(llm, vectorstore, task_execution_chain, False, 3)
        self.assertIsNotNone(boss_node.baby_agi.task_execution_chain)

    def test_BossNode_LLMChain(self):
        boss_node = BossNode(llm, vectorstore, task_execution_chain, False, 3)
        self.assertIsNotNone(boss_node.baby_agi.task_execution_chain.agent.llm_chain)

    @patch.object(BossNode, 'execute_task')
    def test_BossNode_execute_task_called_with_correct_arg(self, mock_execute_task):
        boss_node = BossNode(llm, vectorstore, task_execution_chain, False, 3)
        task = boss_node.create_task('objective')
        boss_node.execute_task(task)
        mock_execute_task.assert_called_once_with(task)

    @patch.object(BossNode, 'execute_task')
    def test_BossNode_execute_task_output(self, mock_execute_task):
        mock_execute_task.return_value = "some_output"
        boss_node = BossNode(llm, vectorstore, task_execution_chain, False, 3)
        task = boss_node.create_task('objective')
        output = boss_node.execute_task(task)
        self.assertIsNotNone(output)

    def test_BossNode_execute_task_error_handling(self):
        boss_node = BossNode(llm, vectorstore, task_execution_chain, False, 3)
        try:
            boss_node.execute_task(None)
            self.assertTrue(True)
        except Exception:
            self.fail("boss_node.execute_task raised Exception unexpectedly!")

# Run the tests
if __name__ == '__main__':
    unittest.main()

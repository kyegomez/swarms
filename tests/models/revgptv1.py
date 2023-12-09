import unittest
from unittest.mock import patch
from Sswarms.models.revgptv1 import RevChatGPTModelv1


class TestRevChatGPT(unittest.TestCase):
    def setUp(self):
        self.access_token = "<your_access_token>"
        self.model = RevChatGPTModelv1(access_token=self.access_token)

    def test_run(self):
        prompt = "What is the capital of France?"
        response = self.model.run(prompt)
        self.assertEqual(response, "The capital of France is Paris.")

    def test_run_time(self):
        prompt = "Generate a 300 word essay about technology."
        self.model.run(prompt)
        self.assertLess(self.model.end_time - self.model.start_time, 60)

    def test_generate_summary(self):
        text = "This is a sample text to summarize. It has multiple sentences and details. The summary should be concise."
        summary = self.model.generate_summary(text)
        self.assertLess(len(summary), len(text) / 2)

    def test_enable_plugin(self):
        plugin_id = "some_plugin_id"
        self.model.enable_plugin(plugin_id)
        self.assertIn(plugin_id, self.model.config["plugin_ids"])

    def test_list_plugins(self):
        plugins = self.model.list_plugins()
        self.assertGreater(len(plugins), 0)
        self.assertIsInstance(plugins[0], dict)
        self.assertIn("id", plugins[0])
        self.assertIn("name", plugins[0])

    def test_get_conversations(self):
        conversations = self.model.chatbot.get_conversations()
        self.assertIsInstance(conversations, list)

    @patch("RevChatGPTModelv1.Chatbot.get_msg_history")
    def test_get_msg_history(self, mock_get_msg_history):
        conversation_id = "convo_id"
        self.model.chatbot.get_msg_history(conversation_id)
        mock_get_msg_history.assert_called_with(conversation_id)

    @patch("RevChatGPTModelv1.Chatbot.share_conversation")
    def test_share_conversation(self, mock_share_conversation):
        self.model.chatbot.share_conversation()
        mock_share_conversation.assert_called()

    def test_gen_title(self):
        convo_id = "123"
        message_id = "456"
        title = self.model.chatbot.gen_title(convo_id, message_id)
        self.assertIsInstance(title, str)

    def test_change_title(self):
        convo_id = "123"
        title = "New Title"
        self.model.chatbot.change_title(convo_id, title)
        self.assertEqual(self.model.chatbot.get_msg_history(convo_id)["title"], title)

    def test_delete_conversation(self):
        convo_id = "123"
        self.model.chatbot.delete_conversation(convo_id)
        with self.assertRaises(Exception):
            self.model.chatbot.get_msg_history(convo_id)

    def test_clear_conversations(self):
        self.model.chatbot.clear_conversations()
        conversations = self.model.chatbot.get_conversations()
        self.assertEqual(len(conversations), 0)

    def test_rollback_conversation(self):
        original_convo_id = self.model.chatbot.conversation_id
        self.model.chatbot.rollback_conversation(1)
        self.assertNotEqual(original_convo_id, self.model.chatbot.conversation_id)


if __name__ == "__main__":
    unittest.main()

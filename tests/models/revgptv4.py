import unittest
from unittest.mock import patch
from RevChatGPTModelv4 import RevChatGPTModelv4


class TestRevChatGPT(unittest.TestCase):
    def setUp(self):
        self.access_token = "123"
        self.model = RevChatGPTModelv4(access_token=self.access_token)

    def test_run(self):
        prompt = "What is the capital of France?"
        self.model.start_time = 10
        self.model.end_time = 20
        response = self.model.run(prompt)
        self.assertEqual(response, "The capital of France is Paris.")
        self.assertEqual(self.model.start_time, 10)
        self.assertEqual(self.model.end_time, 20)

    def test_generate_summary(self):
        text = "Hello world. This is some text. It has multiple sentences."
        summary = self.model.generate_summary(text)
        self.assertEqual(summary, "")

    @patch("RevChatGPTModelv4.Chatbot.install_plugin")
    def test_enable_plugin(self, mock_install_plugin):
        plugin_id = "plugin123"
        self.model.enable_plugin(plugin_id)
        mock_install_plugin.assert_called_with(plugin_id=plugin_id)

    @patch("RevChatGPTModelv4.Chatbot.get_plugins")
    def test_list_plugins(self, mock_get_plugins):
        mock_get_plugins.return_value = [{"id": "123", "name": "Test Plugin"}]
        plugins = self.model.list_plugins()
        self.assertEqual(len(plugins), 1)
        self.assertEqual(plugins[0]["id"], "123")
        self.assertEqual(plugins[0]["name"], "Test Plugin")

    @patch("RevChatGPTModelv4.Chatbot.get_conversations")
    def test_get_conversations(self, mock_get_conversations):
        self.model.chatbot.get_conversations()
        mock_get_conversations.assert_called()

    @patch("RevChatGPTModelv4.Chatbot.get_msg_history")
    def test_get_msg_history(self, mock_get_msg_history):
        convo_id = "123"
        self.model.chatbot.get_msg_history(convo_id)
        mock_get_msg_history.assert_called_with(convo_id)

    @patch("RevChatGPTModelv4.Chatbot.share_conversation")
    def test_share_conversation(self, mock_share_conversation):
        self.model.chatbot.share_conversation()
        mock_share_conversation.assert_called()

    @patch("RevChatGPTModelv4.Chatbot.gen_title")
    def test_gen_title(self, mock_gen_title):
        convo_id = "123"
        message_id = "456"
        self.model.chatbot.gen_title(convo_id, message_id)
        mock_gen_title.assert_called_with(convo_id, message_id)

    @patch("RevChatGPTModelv4.Chatbot.change_title")
    def test_change_title(self, mock_change_title):
        convo_id = "123"
        title = "New Title"
        self.model.chatbot.change_title(convo_id, title)
        mock_change_title.assert_called_with(convo_id, title)

    @patch("RevChatGPTModelv4.Chatbot.delete_conversation")
    def test_delete_conversation(self, mock_delete_conversation):
        convo_id = "123"
        self.model.chatbot.delete_conversation(convo_id)
        mock_delete_conversation.assert_called_with(convo_id)

    @patch("RevChatGPTModelv4.Chatbot.clear_conversations")
    def test_clear_conversations(self, mock_clear_conversations):
        self.model.chatbot.clear_conversations()
        mock_clear_conversations.assert_called()

    @patch("RevChatGPTModelv4.Chatbot.rollback_conversation")
    def test_rollback_conversation(self, mock_rollback_conversation):
        num = 2
        self.model.chatbot.rollback_conversation(num)
        mock_rollback_conversation.assert_called_with(num)

    @patch("RevChatGPTModelv4.Chatbot.reset_chat")
    def test_reset_chat(self, mock_reset_chat):
        self.model.chatbot.reset_chat()
        mock_reset_chat.assert_called()


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch, Mock, MagicMock
from apps.discord import Bot  # Replace 'Bot' with the name of the file containing your bot's code.

class TestBot(unittest.TestCase):

    def setUp(self):
        self.llm_mock = Mock()
        self.agent_mock = Mock()
        self.bot = Bot(agent=self.agent_mock, llm=self.llm_mock)

    @patch('Bot.load_dotenv')  # Mocking the `load_dotenv` function call.
    def test_initialization(self, mock_load_dotenv):
        self.assertIsNotNone(self.bot.bot)
        self.assertEqual(self.bot.agent, self.agent_mock)
        self.assertEqual(self.bot.llm, self.llm_mock)
        mock_load_dotenv.assert_called_once()

    @patch('Bot.commands.bot')
    def test_greet(self, mock_bot):
        ctx_mock = Mock()
        ctx_mock.author.name = "TestUser"
        self.bot.bot.clear()
        self.bot.bot.greet(ctx_mock)
        ctx_mock.send.assert_called_with("hello, TestUser!")

    # Similarly, you can add tests for other commands.

    @patch('Bot.commands.bot')
    def test_help_me(self, mock_bot):
        ctx_mock = Mock()
        self.bot.bot.clear()
        self.bot.bot.help_me(ctx_mock)
        # Verify the help text was sent. You can check for a substring to make it shorter.
        ctx_mock.send.assert_called()

    @patch('Bot.commands.bot')
    def test_on_command_error(self, mock_bot):
        ctx_mock = Mock()
        error_mock = Mock()
        error_mock.__class__.__name__ = "CommandNotFound"
        self.bot.bot.clear()
        self.bot.bot.on_command_error(ctx_mock, error_mock)
        ctx_mock.send.assert_called_with("that command does not exist!")

    def test_add_command(self):
        def sample_function(*args):
            return "Test Response"

        self.bot.add_command("test_command", sample_function)
        # Here, you can further test by triggering the command and checking the response.

    # You can add more tests for other commands and functionalities.

if __name__ == "__main__":
    unittest.main()

from langchain.tools import tool

from swarms.tools.base import BaseToolSet, SessionGetter, ToolScope
from swarms.utils.logger import logger


class ExitConversation(BaseToolSet):
    @tool(
        name="Exit Conversation",
        description="A tool to exit the conversation. "
        "Use this when you want to exit the conversation. "
        "The input should be a message that the conversation is over.",
        scope=ToolScope.SESSION,
    )
    def exit(self, message: str, get_session: SessionGetter) -> str:
        """Run the tool."""
        _, executor = get_session()
        del executor

        logger.debug("\nProcessed ExitConversation.")

        return message
    



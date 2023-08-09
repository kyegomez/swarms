from swarms.agents.tools.base import BaseToolSet, SessionGetter, ToolScope, Tool
from swarms.utils.logger import logger


class ExitConversation(BaseToolSet):
    @Tool(
        name="Exit Conversation",
        description="A tool to exit the conversation. "
        "Use this when you want to exit the conversation. "
        "The input should be a message that the conversation is over.",
        # scope=ToolScope.SESSION,
    )
    def exit(self, message: str, get_session: SessionGetter) -> str:
        """Run the tool."""
        _, executor = get_session()
        del executor

        logger.debug("\nProcessed ExitConversation.")

        return message
    



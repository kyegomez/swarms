from langchain.models import Anthropic, GooglePalm, OpenAIChat
from swarms.swarms import GodMode

claude = Anthropic(anthropic_api_key="")
palm = GooglePalm(google_api_key="")
gpt = OpenAIChat(openai_api_key="")

# Usage
llms = [claude, palm, gpt]

god_mode = GodMode(llms)

task = "What are the biggest risks facing humanity?"

god_mode.print_responses(task)

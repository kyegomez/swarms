
from swarms.models import Anthropic, GooglePalm, OpenAIChat
from swarms.swarms.god_mode import GodMode

claude = Anthropic(anthropic_api_key="")
palm = GooglePalm(google_api_key="")
gpt = OpenAIChat(openai_api_key="")

# Usage
llms = [
   claude,
   palm,
   gpt 
]

god_mode = GodMode(llms)

task = f"What are the biggest risks facing humanity?"

god_mode.print_responses(task)
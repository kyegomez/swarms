from swarms import GodMode

# Usage
llms = [Anthropic(model="<model_name>", anthropic_api_key="my-api-key") for _ in range(5)]

god_mode = GodMode(llms)

task = f"{anthropic.HUMAN_PROMPT} What are the biggest risks facing humanity?{anthropic.AI_PROMPT}"

god_mode.print_responses(task)
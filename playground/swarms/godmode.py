from swarms.swarms import ModelParallelizer
from swarms.models import OpenAIChat

api_key = ""

llm = OpenAIChat(openai_api_key=api_key)


llms = [llm, llm, llm]

god_mode = ModelParallelizer(llms)

task = "Generate a 10,000 word blog on health and wellness."

out = god_mode.run(task)
god_mode.print_responses(task)

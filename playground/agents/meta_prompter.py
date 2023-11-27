from swarms.workers import Worker
from swarms.agents.meta_prompter import MetaPrompterAgent
from swarms.models import OpenAI

# init llm
llm = OpenAI()

# init the meta prompter agent that optimized prompts
meta_optimizer = MetaPrompterAgent(llm=llm)

# init the worker agent
worker = Worker(llm)

# broad task to complete
task = "Create a feedforward in pytorch"

# optimize the prompt
optimized_prompt = meta_optimizer.run(task)

# run the optimized prompt with detailed instructions
result = worker.run(optimized_prompt)

# print
print(result)

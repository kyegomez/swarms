from swarms.workers import Worker
from swarms.agents import MetaPrompterAgent
from langchain.llms import OpenAI

llm = OpenAI()

task = "Create a feedforward in pytorch"
agent = MetaPrompterAgent(llm=llm)
optimized_prompt = agent.run(task)    

worker = Worker(llm)
worker.run(optimized_prompt)
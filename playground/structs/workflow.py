from swarms.structs.workflow import Workflow, StringTask
from langchain.llms import OpenAIChat


llm = OpenAIChat()


workflow = Workflow(llm)

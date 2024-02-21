from swarms.models import OpenAIChat
from swarms.structs.workflow import Workflow

llm = OpenAIChat()


workflow = Workflow(llm)

from swarms.structs.workflow import Workflow
from swarms.models import OpenAIChat
import os

api_key = os.getenv("OPENAI_API_KEY")
org_id = os.getenv("OPENAI_ORG_ID")

llm = OpenAIChat(openai_api_key=api_key, openai_org_id=org_id))


workflow = Workflow(llm)

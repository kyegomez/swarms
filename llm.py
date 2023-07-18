# example
from swarms.utils.llm import LLM
llm_instance = LLM(hf_repo_id="google/flan-t5-xl", hf_api_token="your_hf_api_token")
result = llm_instance.run("Who won the FIFA World Cup in 1998?")
print(result)
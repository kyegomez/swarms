from swarms.utils.llm import LLM

# using HuggingFaceHub
llm_instance = LLM(hf_repo_id="Writer/camel-5b-hf", hf_api_token="your_hf_api_token")
result = llm_instance.run("Who won the FIFA World Cup in 1998?")
print(llm_instance)

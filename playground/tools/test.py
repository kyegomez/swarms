from swarms.tools.singletool import load_single_tools, STQuestionAnswerer

# Langchain
tool_name, tool_url = "weather", "http://127.0.0.1:8079/tools/weather/"
tool_name, tool_config = load_single_tools(tool_name, tool_url)
print(tool_name, tool_config)
stqa = STQuestionAnswerer()

agent = stqa.load_tools(
    tool_name, tool_config, prompt_type="react-with-tool-description"
)
agent("write a weather report for SF today")

# BabyAGI
# tool_name, tool_url = 'weather',  "http://127.0.0.1:8079/tools/weather/"
# tool_name, tool_config = load_single_tools(tool_name, tool_url)
# print(tool_name, tool_config)
# stqa =  STQuestionAnswerer()

# agent = stqa.load_tools(tool_name, tool_config, prompt_type="babyagi")
# agent("write a weather report for SF today")

# Auto-GPT
# tool_name, tool_url = 'weather',  "http://127.0.0.1:8079/tools/weather/"
# tool_name, tool_config = load_single_tools(tool_name, tool_url)
# print(tool_name, tool_config)
# stqa =  STQuestionAnswerer()

# agent = stqa.load_tools(tool_name, tool_config, prompt_type="autogpt")
# agent.run(["write a weather report for SF today"])


""" 
from swarms.agent.singletool import load_single_tools, STQuestionAnswerer

tool_name, tool_url = 'wikipedia',  "http://127.0.0.1:8079/tools/wikipedia/"
tool_name, tool_config = load_single_tools(tool_name, tool_url)
print(tool_name, tool_config)
stqa =  STQuestionAnswerer()

agent = stqa.load_tools(tool_name, tool_config, prompt_type="babyagi")
# agent = stqa.load_tools(tool_name, tool_config, prompt_type="react-with-tool-description")# prompt_type="babyagi")
agent("Where is Yaoming Born?")
"""

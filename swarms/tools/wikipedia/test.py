from swarms.tools.agent.singletool import load_single_tools, STQuestionAnswerer
import requests
import json

# at = "{\"entity\": \"Arthur\"s Magazine\"}"
# print(at[19])
# print(len(at))
# a = json.loads("{\"entity\": \"Arthur\"s Magazine\"}")
# print(a)

tool_name, tool_url = "wikipedia", "http://127.0.0.1:8079/tools/wikipedia/"
tools_name, tools_config = load_single_tools(tool_name, tool_url)
print(tools_name, tools_config)

qa = STQuestionAnswerer()

agent = qa.load_tools(tools_name, tools_config)

agent("Which magazine was started first, Arthur’s Magazine or First for Women?")

agent("when was the first hunger games book published?")

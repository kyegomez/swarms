from swarms.tools.agent.singletool import load_single_tools, STQuestionAnswerer

tool_name, tool_url = "Airbnb", "http://127.0.0.1:8079/tools/airbnb/"
tools_name, tools_config = load_single_tools(tool_name, tool_url)
print(tools_name, tools_config)

qa = STQuestionAnswerer()

agent = qa.load_tools(tools_name, tools_config)

agent("List some houses to rent in Santa Monica, CA.")

from swarms.tools.agent.singletool import load_single_tools, STQuestionAnswerer

tool_name, tool_url = "google_places", "http://127.0.0.1:8079/tools/google_places/"
tools_name, tools_config = load_single_tools(tool_name, tool_url)
print(tools_name, tools_config)

qa = STQuestionAnswerer()

agent = qa.load_tools(tools_name, tools_config)

agent("Where is Tsinghua University?")
agent("List a few locations of KFC in Beijing.")

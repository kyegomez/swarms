from swarms.tools.agent.singletool import load_single_tools, STQuestionAnswerer

tool_name, tool_url = "google_serper", "http://127.0.0.1:8079/tools/google_serper/"
tools_name, tools_config = load_single_tools(tool_name, tool_url)
print(tools_name, tools_config)

qa = STQuestionAnswerer()

agent = qa.load_tools(tools_name, tools_config)

agent("Where was the current US President born?")
agent("List 5 well-known Chinese restaurants in New York and their location.")

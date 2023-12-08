from swarms.tools.agent.singletool import load_single_tools, STQuestionAnswerer

tool_name, tool_url = "file_operation", "http://127.0.0.1:8079/tools/file_operation/"
tool_name, tool_config = load_single_tools(tool_name, tool_url)
print(tool_name, tool_config)
stqa = STQuestionAnswerer()

agent = stqa.load_tools(tool_name, tool_config)
agent("write hello world to test.txt")

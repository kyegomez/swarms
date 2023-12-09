from swarms.tools.agent.singletool import load_single_tools, STQuestionAnswerer

tool_name, tool_url = "wolframalpha", "http://127.0.0.1:8079/tools/wolframalpha/"
tools_name, tools_config = load_single_tools(tool_name, tool_url)
print(tools_name, tools_config)

qa = STQuestionAnswerer()

agent = qa.load_tools(tools_name, tools_config)

agent("Calc integral of sin(x)+2x^2+3x+1 from 0 to 1")

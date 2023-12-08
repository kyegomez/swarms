from swarms.tools.agent.singletool import load_single_tools, STQuestionAnswerer

tool_name, tool_url = "sceneXplain", "http://127.0.0.1:8079/tools/sceneXplain/"
tools_name, tools_config = load_single_tools(tool_name, tool_url)
print(tools_name, tools_config)

qa = STQuestionAnswerer()

agent = qa.load_tools(tools_name, tools_config)

agent(
    "What is in this image https://storage.googleapis.com/causal-diffusion.appspot.com/imagePrompts%2F0rw369i5h9t%2Foriginal.png?"
)
agent("Describe Van Gogh's painting The Starry Night.")

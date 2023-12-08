from swarms.tools.agent.singletool import load_single_tools, STQuestionAnswerer

tool_name, tool_url = "meta_analysis", "http://127.0.0.1:8079/tools/meta_analysis/"
tools_name, tools_config = load_single_tools(tool_name, tool_url)
print(tools_name, tools_config)

qa = STQuestionAnswerer()

agent = qa.load_tools(tools_name, tools_config)

agent(
    "How's the treatment for COVID-19 of Lianhua Qingwen? Answer this question by analyzing literatures that meet this criteria: trials are conducted for comparing Lianhua Qingwen with other conventional treatments for Coronavirus disease. Patients are adults with COVID-19 diagnosis."
)
agent(
    "Recommend some literatures about trials of treating COVID-19 with Lianhua Qingwen for me. Print their title/abstract. "
)
agent.run(
    [
        "Help me analyze trials of Lianhua Qingwen for treating COVID-19, and Paxlovid for treating COVID-19 separately, and then compare them."
    ]
)

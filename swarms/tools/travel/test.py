from swarms.tools.agent.singletool import load_single_tools, STQuestionAnswerer
from datetime import datetime, timedelta

tool_name, tool_url = "travel", "http://127.0.0.1:8079/tools/travel/"
tools_name, tools_config = load_single_tools(tool_name, tool_url)
print(tools_name, tools_config)

qa = STQuestionAnswerer()

agent = qa.load_tools(tools_name, tools_config)

month_later = str(datetime.now() + timedelta(days=30)).split()[0]
agent(
    f"I will go to Seattle from Beijing on {month_later}. Can you make a recommendation on hotels and flight please?"
)
agent("Please help me rent a car near Tsinghua University.")
agent(
    "Where can I visit on weekend if I am going to University of Washington this summer?"
)

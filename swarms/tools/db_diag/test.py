from swarms.tools.agent.singletool import load_single_tools, STQuestionAnswerer
import datetime

tool_name, tool_url = "db_diag", "http://127.0.0.1:8079/tools/db_diag/"
tool_name, tool_config = load_single_tools(tool_name, tool_url)
print(tool_name, tool_config)
stqa = STQuestionAnswerer()

# langchain
agent = stqa.load_tools(
    tool_name, tool_config, prompt_type="react-with-tool-description"
)  # langchain: react-with-tool-description autogpt: autogpt

# database on 123.56.63.105

"""
start_timestamp_str = "2023-05-19 22:21:30"
dt = datetime.datetime.strptime(start_timestamp_str, "%Y-%m-%d %H:%M:%S")
timestamp = dt.timestamp()
start_time = timestamp

end_timestamp_str = "2023-05-19 22:23:30"
dt = datetime.datetime.strptime(end_timestamp_str, "%Y-%m-%d %H:%M:%S")
timestamp = dt.timestamp()
end_time = timestamp

print(" ===== time period: ", start_time, end_time)
"""

# text = "The database performance is bad during {} to {}.".format(start_timestamp_str, end_timestamp_str) # trigger database diagnosis

text = "Here is a database performance problem. Please help me to diagnose the causes and give some optimization suggestions."

agent(
    """ {}

First, obtain_start_and_end_time_of_anomaly and memorize the start and end time of the anomaly.

Second, you need to diagnose the causes of the anomaly from the following two aspects:

    - call the whether_is_abnormal_metric API and examine whether CPU usage is high (or abnormal). Next, if the CPU usage is high (or abnormal), cpu_diagnosis_agent and obtain the diagnosis results.

    - call the whether_is_abnormal_metric API and examine whether memory usage is high (or abnormal). Next, if the memory usage is high (or abnormal), memory_diagnosis_agent and obtain the diagnosis results.

Third, you need to match each cause with potential solutions cached in the vector database.

Finally, list the above diagnosed causes and their matched solutions in easy-to-understand format using bullet points.

================================
A Demonstration example:

Thought: I need to check whether the CPU usage is high or abnormal during the given time period.

Action: whether_is_abnormal_metric

Action Input: {{"start_time": xxxx, "end_time": xxxx, "metric_name": "cpu_usage"}}

Note. 1) The first action must be obtain_start_and_end_time_of_anomaly;
2) Do not use any image in the output; 
3) Give some optimization suggestions based on the diagnosis results.
""".format(
        text
    )
)

"""
1) Action can only be one of the following API names: obtain_start_and_end_time_of_anomaly, whether_is_abnormal_metric, obtain_values_of_metrics, cpu_diagnosis_agent, memory_diagnosis_agent. Any other content in Action is unacceptable;
"""

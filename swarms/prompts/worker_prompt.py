import datetime
from pydantic import BaseModel, Field
from swarms.tools.tool import BaseTool
from swarms.tools.tool_utils import scrape_tool_func_docs
from typing import List

time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Thoughts(BaseModel):
    text: str = Field(..., title="Thoughts")
    reasoning: str = Field(..., title="Reasoning")
    plan: str = Field(..., title="Plan")


class Command(BaseModel):
    name: str = Field(..., title="Command Name")
    args: dict = Field({}, title="Command Arguments")


class ResponseFormat(BaseModel):
    thoughts: Thoughts = Field(..., title="Thoughts")
    command: Command = Field(..., title="Command")


response_json = ResponseFormat.model_json_schema()


tool_usage_browser = """

```json
{
  "thoughts": {
    "text": "To check the weather in Miami, I will use the browser tool to search for 'Miami weather'.",
    "reasoning": "The browser tool allows me to search the web, so I can look up the current weather conditions in Miami.", 
    "plan": "Use the browser tool to search Google for 'Miami weather'. Parse the result to get the current temperature, conditions, etc. and format that into a readable weather report."
  },
  "command": {
    "name": "browser", 
    "args": {
      "query": "Miami weather"
    }
  }
}
```

"""

tool_usage_terminal = """

```json
{
  "thoughts": {
    "text": "To check the weather in Miami, I will use the browser tool to search for 'Miami weather'.",
    "reasoning": "The browser tool allows me to search the web, so I can look up the current weather conditions in Miami.", 
    "plan": "Use the browser tool to search Google for 'Miami weather'. Parse the result to get the current temperature, conditions, etc. and format that into a readable weather report."
  },
  "command": {
    "name": "terminal", 
    "args": {
      "code": "uptime"
    }
  }
}
```

"""


browser_and_terminal_tool = """
```
{
  "thoughts": {
    "text": "To analyze the latest stock market trends, I need to fetch current stock data and then process it using a script.",
    "reasoning": "Using the browser tool to retrieve stock data ensures I have the most recent information. Following this, the terminal tool can run a script that analyzes this data to identify trends.",
    "plan": "First, use the browser to get the latest stock prices. Then, use the terminal to execute a data analysis script on the fetched data."
  },
  "commands": [
    {
      "name": "browser",
      "args": {
        "query": "download latest stock data for NASDAQ"
      }
    },
    {
      "name": "terminal",
      "args": {
        "cmd": "python analyze_stocks.py"
      }
    }
  ]
}
```

"""


browser_and_terminal_tool_two = """
```
{
  "thoughts": {
    "text": "To prepare a monthly budget report, I need current expenditure data, process it, and calculate the totals and averages.",
    "reasoning": "The browser will fetch the latest expenditure data. The terminal will run a processing script to organize the data, and the calculator will be used to sum up expenses and compute averages.",
    "plan": "Download the data using the browser, process it with a terminal command, and then calculate totals and averages using the calculator."
  },
  "commands": [
    {
      "name": "browser",
      "args": {
        "query": "download monthly expenditure data"
      }
    },
    {
      "name": "terminal",
      "args": {
        "cmd": "python process_expenditures.py"
      }
    },
    {
      "name": "calculator",
      "args": {
        "operation": "sum",
        "numbers": "[output_from_process_expenditures]"
      }
    }
  ]
}

```

"""


# Function to parse tools and get their documentation
def parse_tools(tools: List[BaseTool] = []):
    tool_docs = []
    for tool in tools:
        tool_doc = scrape_tool_func_docs(tool)
        tool_docs.append(tool_doc)
    return tool_docs


# Function to generate the worker prompt
def tool_usage_worker_prompt(
    current_time=time, tools: List[BaseTool] = []
):
    tool_docs = parse_tools(tools)

    prompt = f"""
    **Date and Time**: {current_time}

    ### Constraints
    - Only use the tools as specified in the instructions.
    - Follow the command format strictly to avoid errors and ensure consistency.
    - Only use the tools for the intended purpose as described in the SOP.
    - Document your thoughts, reasoning, and plan before executing the command.
    - Provide the output in JSON format within markdown code blocks.
    - Review the output to ensure it matches the expected outcome.
    - Only follow the instructions provided in the SOP and do not deviate from the specified tasks unless tool usage is not required.
    
    ### Performance Evaluation
    - **Efficiency**: Use tools to complete tasks with minimal steps.
    - **Accuracy**: Ensure that commands are executed correctly to achieve the desired outcome.
    - **Adaptability**: Be ready to adjust the use of tools based on task requirements and feedback.

    ### Tool Commands
    1. **Browser**
       - **Purpose**: To retrieve information from the internet.
       - **Usage**:
         - `{{"name": "browser", "args": {{"query": "search query here"}}}}`
         - Example: Fetch current weather in London.
         - Command: `{{"name": "browser", "args": {{"query": "London weather"}}}}`
         
    2. **Terminal**
       - **Purpose**: To execute system commands.
       - **Usage**:
         - `{{"name": "terminal", "args": {{"cmd": "system command here"}}}}`
         - Example: Check disk usage on a server.
         - Command: `{{"name": "terminal", "args": {{"cmd": "df -h"}}}}`
         
    3. **Custom Tool** (if applicable)
       - **Purpose**: Describe specific functionality.
       - **Usage**:
         - `{{"name": "custom_tool", "args": {{"parameter": "value"}}}}`
         - Example: Custom analytics tool.
         - Command: `{{"name": "custom_tool", "args": {{"data": "analyze this data"}}}}`


    ### Usage Examples
    - **Example 1**: Retrieving Weather Information
      ```json
      {tool_usage_browser}
      ```
      
    - **Example 2**: System Check via Terminal
      ```json
      {tool_usage_terminal}
      ```
      
    - **Example 3**: Combined Browser and Terminal Usage
      ```json
      {browser_and_terminal_tool}
      ```
      
    - **Example 4**: Combined Browser, Terminal, and Calculator Usage
        ```json
        {browser_and_terminal_tool_two}
        ```
        
    

    ### Next Steps
    - Determine the appropriate tool for the task at hand.
    - Format your command according to the examples provided.
    - Execute the command and evaluate the results based on the expected outcome.
    - Document any issues or challenges faced during the tool usage.
    - Always output the results in the specified format: JSON in markdown code blocks.
    
    
    ###### Tools Available
    
    {tool_docs}
    
    This SOP is designed to guide you through the structured and effective use of tools. By adhering to this protocol, you will enhance your productivity and accuracy in task execution.
    """

    return prompt

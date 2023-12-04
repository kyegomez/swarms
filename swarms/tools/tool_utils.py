import re
import json


def extract_tool_commands(self, text: str):
    """
    Extract the tool commands from the text

    Example:
    ```json
    {
        "tool": "tool_name",
        "params": {
            "tool1": "inputs",
            "param2": "value2"
        }
    }
    ```

    """
    # Regex to find JSON like strings
    pattern = r"```json(.+?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    json_commands = []
    for match in matches:
        try:
            json_commands = json.loads(match)
            json_commands.append(json_commands)
        except Exception as error:
            print(f"Error parsing JSON command: {error}")


def parse_and_execute_tools(response: str):
    """Parse and execute the tools"""
    json_commands = extract_tool_commands(response)
    for command in json_commands:
        tool_name = command.get("tool")
        params = command.get("parmas", {})
        execute_tools(tool_name, params)


def execute_tools(self, tool_name, params):
    """Execute the tool with the provided params"""
    tool = self.tool_find_by_name(tool_name)
    if tool:
        # Execute the tool with the provided parameters
        tool_result = tool.run(**params)
        print(tool_result)

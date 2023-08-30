#props to shroominic
from swarms.tools.base import Tool, ToolException
from typing import Callable, Any, List
from codeinterpreterapi import CodeInterpreterSession, File, ToolException

class CodeInterpreter(Tool):
    def __init__(self, name: str, description: str):
        super().__init__(name, description, self.run)

    def run(self, user_request: str, file_paths: List[str] = []) -> Any:
        # create a session
        session = CodeInterpreterSession()
        session.start()

        # create files from paths
        files = [File.from_path(file_path) for file_path in file_paths]

        try:
            # generate a response based on user input
            response = session.generate_response(user_request, files=files)

            # output the response (text + image)
            print("AI: ", response.content)
            for file in response.files:
                file.show_image()
        except Exception as e:
            raise ToolException(f"Error running CodeInterpreter: {e}")
        finally:
            # terminate the session
            session.stop()

    async def arun(self, user_request: str, file_paths: List[str] = []) -> Any:
        # create a session
        session = CodeInterpreterSession()
        await session.astart()

        # create files from paths
        files = [File.from_path(file_path) for file_path in file_paths]

        try:
            # generate a response based on user input
            response = await session.generate_response(user_request, files=files)

            # output the response (text + image)
            print("AI: ", response.content)
            for file in response.files:
                file.show_image()
        except Exception as e:
            raise ToolException(f"Error running CodeInterpreter: {e}")
        finally:
            # terminate the session
            await session.astop()

"""

tool = CodeInterpreter("Code Interpreter", "A tool to interpret code and generate useful outputs.")
tool.run("Plot the bitcoin chart of 2023 YTD")

# Or with file inputs
tool.run("Analyze this dataset and plot something interesting about it.", ["examples/assets/iris.csv"])



import asyncio

tool = CodeInterpreter("Code Interpreter", "A tool to interpret code and generate useful outputs.")
asyncio.run(tool.arun("Plot the bitcoin chart of 2023 YTD"))

# Or with file inputs
asyncio.run(tool.arun("Analyze this dataset and plot something interesting about it.", ["examples/assets/iris.csv"]))
"""
import os
from enum import Enum
from typing import Callable, Tuple

from langchain.agents.agent import AgentExecutor
from langchain.agents.tools import BaseTool, Tool
from typing import Optional

from langchain.agents import load_tools
from langchain.agents.tools import BaseTool
from langchain.llms.base import BaseLLM



import requests
from bs4 import BeautifulSoup

from swarms.utils.logger import logger
class ToolScope(Enum):
    GLOBAL = "global"
    SESSION = "session"


SessionGetter = Callable[[], Tuple[str, AgentExecutor]]

def tool(
    name: str,
    description: str,
    scope: ToolScope = ToolScope.GLOBAL,
):
    def decorator(func):
        func.name = name
        func.description = description
        func.is_tool = True
        func.scope = scope
        return func

    return decorator


class ToolWrapper:
    def __init__(self, name: str, description: str, scope: ToolScope, func):
        self.name = name
        self.description = description
        self.scope = scope
        self.func = func

    def is_global(self) -> bool:
        return self.scope == ToolScope.GLOBAL

    def is_per_session(self) -> bool:
        return self.scope == ToolScope.SESSION

    def to_tool(
        self,
        get_session: SessionGetter = lambda: [],
    ) -> BaseTool:
        func = self.func
        if self.is_per_session():
            def func(*args, **kwargs):
                return self.func(*args, **kwargs, get_session=get_session)

        return Tool(
            name=self.name,
            description=self.description,
            func=func,
        )


class BaseToolSet:
    def tool_wrappers(cls) -> list[ToolWrapper]:
        methods = [
            getattr(cls, m) for m in dir(cls) if hasattr(getattr(cls, m), "is_tool")
        ]
        return [ToolWrapper(m.name, m.description, m.scope, m) for m in methods]
    



class RequestsGet(BaseToolSet):
    # @tool(
    #     name="Requests Get",
    #     description="A portal to the internet. "
    #     "Use this when you need to get specific content from a website."
    #     "Input should be a  url (i.e. https://www.google.com)."
    #     "The output will be the text response of the GET request.",
    # )
    def get(self, url: str) -> str:
        """Run the tool."""
        html = requests.get(url).text
        soup = BeautifulSoup(html)
        non_readable_tags = soup.find_all(
            ["script", "style", "header", "footer", "form"]
        )

        for non_readable_tag in non_readable_tags:
            non_readable_tag.extract()

        content = soup.get_text("\n", strip=True)

        if len(content) > 300:
            content = content[:300] + "..."

        logger.debug(
            f"\nProcessed RequestsGet, Input Url: {url} " f"Output Contents: {content}"
        )

        return content


# class WineDB(BaseToolSet):
#     def __init__(self):
#         db = DatabaseReader(
#             scheme="postgresql",  # Database Scheme
#             host=settings["WINEDB_HOST"],  # Database Host
#             port="5432",  # Database Port
#             user="alphadom",  # Database User
#             password=settings["WINEDB_PASSWORD"],  # Database Password
#             dbname="postgres",  # Database Name
#         )
#         self.columns = ["nameEn", "nameKo", "description"]
#         concat_columns = str(",'-',".join([f'"{i}"' for i in self.columns]))
#         query = f"""
#             SELECT
#                 Concat({concat_columns})
#             FROM wine
#         """
#         documents = db.load_data(query=query)
#         self.index = GPTVectorStoreIndex(documents)

#     @tool(
#         name="Wine Recommendation",
#         description="A tool to recommend wines based on a user's input. "
#         "Inputs are necessary factors for wine recommendations, such as the user's mood today, side dishes to eat with wine, people to drink wine with, what things you want to do, the scent and taste of their favorite wine."
#         "The output will be a list of recommended wines."
#         "The tool is based on a database of wine reviews, which is stored in a database.",
#     )
#     def recommend(self, query: str) -> str:
#         """Run the tool."""
#         results = self.index.query(query)
#         wine = "\n".join(
#             [
#                 f"{i}:{j}"
#                 for i, j in zip(
#                     self.columns, results.source_nodes[0].source_text.split("-")
#                 )
#             ]
#         )
#         output = results.response + "\n\n" + wine

#         logger.debug(
#             f"\nProcessed WineDB, Input Query: {query} " f"Output Wine: {wine}"
#         )

#         return output


class ExitConversation(BaseToolSet):
    @tool(
        name="Exit Conversation",
        description="A tool to exit the conversation. "
        "Use this when you want to exit the conversation. "
        "The input should be a message that the conversation is over.",
        scope=ToolScope.SESSION,
    )
    def exit(self, message: str, get_session: SessionGetter) -> str:
        """Run the tool."""
        _, executor = get_session()
        del executor

        logger.debug("\nProcessed ExitConversation.")

        return message
    




class ToolsFactory:
    @staticmethod
    def from_toolset(
        toolset: BaseToolSet,
        only_global: Optional[bool] = False,
        only_per_session: Optional[bool] = False,
        get_session: SessionGetter = lambda: [],
    ) -> list[BaseTool]:
        tools = []
        for wrapper in toolset.tool_wrappers():
            if only_global and not wrapper.is_global():
                continue
            if only_per_session and not wrapper.is_per_session():
                continue
            tools.append(wrapper.to_tool(get_session=get_session))
        return tools

    @staticmethod
    def create_global_tools(
        toolsets: list[BaseToolSet],
    ) -> list[BaseTool]:
        tools = []
        for toolset in toolsets:
            tools.extend(
                ToolsFactory.from_toolset(
                    toolset=toolset,
                    only_global=True,
                )
            )
        return tools

    @staticmethod
    def create_per_session_tools(
        toolsets: list[BaseToolSet],
        get_session: SessionGetter = lambda: [],
    ) -> list[BaseTool]:
        tools = []
        for toolset in toolsets:
            tools.extend(
                ToolsFactory.from_toolset(
                    toolset=toolset,
                    only_per_session=True,
                    get_session=get_session,
                )
            )
        return tools

    @staticmethod
    def create_global_tools_from_names(
        toolnames: list[str],
        llm: Optional[BaseLLM],
    ) -> list[BaseTool]:
        return load_tools(toolnames, llm=llm)
    
##########################################+> 





# ##########################################+>  SYS
# import signal
# from typing import Optional, Tuple

# from ptrace.debugger import (
#     NewProcessEvent,
#     ProcessExecution,
#     ProcessExit,
#     ProcessSignal,
#     PtraceDebugger,
#     PtraceProcess,
# )
# from ptrace.func_call import FunctionCallOptions
# from ptrace.syscall import PtraceSyscall
# from ptrace.tools import signal_to_exitcode


# class SyscallTimeoutException(Exception):
#     def __init__(self, pid: int, *args) -> None:
#         super().__init__(f"deadline exceeded while waiting syscall for {pid}", *args)


# class SyscallTracer:
#     def __init__(self, pid: int):
#         self.debugger: PtraceDebugger = PtraceDebugger()
#         self.pid: int = pid
#         self.process: PtraceProcess = None

#     def is_waiting(self, syscall: PtraceSyscall) -> bool:
#         if syscall.name.startswith("wait"):
#             return True
#         return False

#     def attach(self):
#         self.process = self.debugger.addProcess(self.pid, False)

#     def detach(self):
#         self.process.detach()
#         self.debugger.quit()

#     def set_timer(self, timeout: int):
#         def handler(signum, frame):
#             raise SyscallTimeoutException(self.process.pid)

#         signal.signal(signal.SIGALRM, handler)
#         signal.alarm(timeout)

#     def reset_timer(self):
#         signal.alarm(0)

#     def wait_syscall_with_timeout(self, timeout: int):
#         self.set_timer(timeout)
#         self.process.waitSyscall()
#         self.reset_timer()

#     def wait_until_stop_or_exit(self) -> Tuple[Optional[int], str]:
#         self.process.syscall()
#         exitcode = None
#         reason = ""
#         while True:
#             if not self.debugger:
#                 break

#             try:
#                 self.wait_syscall_with_timeout(30)
#             except ProcessExit as event:
#                 if event.exitcode is not None:
#                     exitcode = event.exitcode
#                 continue
#             except ProcessSignal as event:
#                 event.process.syscall(event.signum)
#                 exitcode = signal_to_exitcode(event.signum)
#                 reason = event.reason
#                 continue
#             except NewProcessEvent as event:
#                 continue
#             except ProcessExecution as event:
#                 continue
#             except Exception as e:
#                 reason = str(e)
#                 break

#             syscall = self.process.syscall_state.event(
#                 FunctionCallOptions(
#                     write_types=False,
#                     write_argname=False,
#                     string_max_length=300,
#                     replace_socketcall=True,
#                     write_address=False,
#                     max_array_count=20,
#                 )
#             )

#             self.process.syscall()

#             if syscall is None:
#                 continue

#             if syscall.result:
#                 continue

#         self.reset_timer()

#         return exitcode, reason
#     ##########################################+> SYS CALL END



# ############### => st dout.py

# import os
# import time
# import subprocess
# from datetime import datetime
# from typing import Callable, Literal, Optional, Union, Tuple

# PipeType = Union[Literal["stdout"], Literal["stderr"]]


# class StdoutTracer:
#     def __init__(
#         self,
#         process: subprocess.Popen,
#         timeout: int = 30,
#         interval: int = 0.1,
#         on_output: Callable[[PipeType, str], None] = lambda: None,
#     ):
#         self.process: subprocess.Popen = process
#         self.timeout: int = timeout
#         self.interval: int = interval
#         self.last_output: datetime = None
#         self.on_output: Callable[[PipeType, str], None] = on_output

#     def nonblock(self):
#         os.set_blocking(self.process.stdout.fileno(), False)
#         os.set_blocking(self.process.stderr.fileno(), False)

#     def get_output(self, pipe: PipeType) -> str:
#         output = None
#         if pipe == "stdout":
#             output = self.process.stdout.read()
#         elif pipe == "stderr":
#             output = self.process.stderr.read()

#         if output:
#             decoded = output.decode()
#             self.on_output(pipe, decoded)
#             self.last_output = datetime.now()
#             return decoded
#         return ""

#     def last_output_passed(self, seconds: int) -> bool:
#         return (datetime.now() - self.last_output).seconds > seconds

#     def wait_until_stop_or_exit(self) -> Tuple[Optional[int], str]:
#         self.nonblock()
#         self.last_output = datetime.now()
#         output = ""
#         exitcode = None
#         while True:
#             new_stdout = self.get_output("stdout")
#             if new_stdout:
#                 output += new_stdout

#             new_stderr = self.get_output("stderr")
#             if new_stderr:
#                 output += new_stderr

#             if self.process.poll() is not None:
#                 exitcode = self.process.poll()
#                 break

#             if self.last_output_passed(self.timeout):
#                 self.process.kill()
#                 break

#             time.sleep(self.interval)

#         return (exitcode, output)

################## => stdout end

import os
import subprocess
from typing import Dict, List

from swarms.utils.main import ANSI, Color, Style # test

class Terminal(BaseToolSet):
    def __init__(self):
        self.sessions: Dict[str, List[SyscallTracer]] = {}

    @tool(
        name="Terminal",
        description="Executes commands in a terminal."
        "If linux errno occurs, we have to solve the problem with the terminal. "
        "Input must be one valid command. "
        "Output will be any output from running that command.",
        scope=ToolScope.SESSION,
    )
    def execute(self, commands: str, get_session: SessionGetter) -> str:
        session, _ = get_session()

        try:
            process = subprocess.Popen(
                commands,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info(ANSI("Realtime Terminal Output").to(Color.magenta()) + ": ")

            output = ""
            tracer = StdoutTracer(
                process,
                on_output=lambda p, o: logger.info(
                    ANSI(p).to(Style.dim()) + " " + o.strip("\n")
                ),
            )
            exitcode, output = tracer.wait_until_stop_or_exit()
        except Exception as e:
            output = str(e)

        logger.debug(
            f"\nProcessed Terminal, Input Commands: {commands} "
            f"Output Answer: {output}"
        )
        return output


# if __name__ == "__main__":
#     import time

#     o = Terminal().execute(
#         "sleep 1; echo 1; sleep 2; echo 2; sleep 3; echo 3; sleep 10;",
#         lambda: ("", None),
#     )
#     print(o)

#     time.sleep(10)  # see if timer has reset


###################=> EDITOR/VERIFY
from pathlib import Path


def verify(func):
    def wrapper(*args, **kwargs):
        try:
            filepath = args[0].filepath
        except AttributeError:
            raise Exception("This tool doesn't have filepath. Please check your code.")
        if not str(Path(filepath).resolve()).startswith(str(Path().resolve())):
            return "You can't access file outside of playground."
        return func(*args, **kwargs)

    return wrapper
#=====================> EDITOR/END VERIFY


###### EDITOR/WRITE.PY

"""
write protocol:

<filepath>
<content>
"""



class WriteCommand:
    separator = "\n"

    def __init__(self, filepath: str, content: int):
        self.filepath: str = filepath
        self.content: str = content
        self.mode: str = "w"

    def with_mode(self, mode: str) -> "WriteCommand":
        self.mode = mode
        return self

    @verify
    def execute(self) -> str:
        dir_path = os.path.dirname(self.filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(self.filepath, self.mode) as f:
            f.write(self.content)
        return self.content

    @staticmethod
    def from_str(command: str) -> "WriteCommand":
        filepath = command.split(WriteCommand.separator)[0]
        return WriteCommand(filepath, command[len(filepath) + 1 :])


class CodeWriter:
    @staticmethod
    def write(command: str) -> str:
        return WriteCommand.from_str(command).with_mode("w").execute()

    @staticmethod
    def append(command: str) -> str:
        return WriteCommand.from_str(command).with_mode("a").execute()
    
#================> END 



#============================> EDITOR/READ.PY
"""
read protocol:

<filepath>|<start line>-<end line>
"""
class Line:
    def __init__(self, content: str, line_number: int, depth: int):
        self.__content: str = content
        self.__line_number: int = line_number
        self.__depth: int = depth
        self.__children: List[Line] = []

    def get_content(self) -> str:
        return self.__content

    def get_depth(self) -> int:
        return self.__depth

    def append_child(self, child: "Line") -> None:
        self.__children.append(child)

    def find_by_lte_depth(self, depth: int) -> List["Line"]:
        if self.__depth > depth:
            return []

        lines: List[Line] = [self]
        for child in self.__children:
            lines += child.find_by_lte_depth(depth)
        return lines

    def find_by_content(self, content: str) -> List["Line"]:
        if content in self.__content:
            return [self]

        lines: List[Line] = []
        for child in self.__children:
            lines += child.find_by_content(content)
        return lines

    def find_last_lines(self) -> List["Line"]:
        if len(self.__children) == 0:
            return [self]
        else:
            return [self, *self.__children[-1].find_last_lines()]

    def print(self, depth: int = 0) -> None:
        print(f"{'  ' * depth}{self}", end="")
        for child in self.__children:
            child.print(depth + 1)

    def __repr__(self):
        return f"{self.__line_number}: {self.__content}"


class CodeTree:
    def __init__(self):
        self.root: Line = Line("\n", -1, -1)

    def append(self, content: str, line_number: int) -> None:
        last_lines: List[Line] = self.root.find_last_lines()
        new_leading_spaces: int = self.__get_leading_spaces(content)

        previous_line: Line = self.root
        previous_leading_spaces: int = -1
        for line in last_lines:
            leading_spaces = self.__get_leading_spaces(line.get_content())
            if (
                previous_leading_spaces < new_leading_spaces
                and new_leading_spaces <= leading_spaces
            ):
                break
            previous_line, previous_leading_spaces = line, leading_spaces

        new_line_depth: int = previous_line.get_depth() + 1
        previous_line.append_child(Line(content, line_number, new_line_depth))

    def find_from_root(self, depth: int) -> List[Line]:
        return self.root.find_by_lte_depth(depth)

    def find_from_parent(self, depth: int, parent_content: str) -> List[Line]:
        lines: List[Line] = self.root.find_by_content(parent_content)
        if len(lines) == 0:
            return []
        parent = lines[0]
        return parent.find_by_lte_depth(depth + parent.get_depth())

    def print(self):
        print("Code Tree:")
        print("=================================")
        self.root.print()
        print("=================================")

    def __get_leading_spaces(self, content: str) -> int:
        return len(content) - len(content.lstrip())


class ReadCommand:
    separator = "|"

    def __init__(self, filepath: str, start: int, end: int):
        self.filepath: str = filepath
        self.start: int = start
        self.end: int = end

    @verify
    def execute(self) -> str:
        with open(self.filepath, "r") as f:
            code = f.readlines()

        if self.start == self.end:
            code = code[self.start - 1]
        else:
            code = "".join(code[self.start - 1 : self.end])
        return code

    @staticmethod
    def from_str(command: str) -> "ReadCommand":
        filepath, line = command.split(ReadCommand.separator)
        start, end = line.split("-")
        return ReadCommand(filepath, int(start), int(end))


class SummaryCommand:
    separator = "|"

    def __init__(self, filepath: str, depth: int, parent_content: Optional[str] = None):
        self.filepath: str = filepath
        self.depth: int = depth
        self.parent_content: Optional[str] = parent_content

    @verify
    def execute(self) -> str:
        with open(self.filepath, "r") as f:
            code = f.readlines()

        code_tree = CodeTree()
        for i, line in enumerate(code):
            if line.strip() != "":
                code_tree.append(line, i + 1)

        if self.parent_content is None:
            lines = code_tree.find_from_root(self.depth)
        else:
            lines = code_tree.find_from_parent(self.depth, self.parent_content)
        return "".join([str(line) for line in lines])

    @staticmethod
    def from_str(command: str) -> "SummaryCommand":
        command_list: List[str] = command.split(SummaryCommand.separator)
        filepath: str = command_list[0]
        depth: int = int(command_list[1])
        parent_content: str | None = command_list[2] if len(command_list) == 3 else None
        return SummaryCommand(
            filepath=filepath, depth=depth, parent_content=parent_content
        )


class CodeReader:
    @staticmethod
    def read(command: str) -> str:
        return ReadCommand.from_str(command).execute()

    @staticmethod
    def summary(command: str) -> str:
        return SummaryCommand.from_str(command).execute()


# if __name__ == "__main__":
#     summary = CodeReader.summary("read.py|1|class ReadCommand:")
#     print(summary)

#============================> EDITOR/READ.PY END




#=================================> EDITOR/PATCH.PY
"""
patch protocol:

<filepath>|<line>,<col>|<line>,<col>|<content>
---~~~+++===+++~~~---
<filepath>|<line>,<col>|<line>,<col>|<content>
---~~~+++===+++~~~---
...
---~~~+++===+++~~~---

let say original code is:
```
import requests

def crawl_news(keyword):
    url = f"https://www.google.com/search?q={keyword}+news"
    response = requests.get(url)

    news = []
    for result in response:
        news.append(result.text)

    return news
```

and we want to change it to:
```
import requests
from bs4 import BeautifulSoup

def crawl_news(keyword):
    url = f"https://www.google.com/search?q={keyword}+news"
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    news_results = soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd")

    news_titles = []
    for result in news_results:
        news_titles.append(result.text)

    return news_titles
```

then the command will be:
test.py|2,1|2,1|from bs4 import BeautifulSoup

---~~~+++===+++~~~---
test.py|5,5|5,33|html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    news_results = soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd")
---~~~+++===+++~~~---
test.py|7,5|9,13|news_titles = []
    for result in news_results:
        news_titles
---~~~+++===+++~~~---
test.py|11,16|11,16|_titles
"""

import re



class Position:
    separator = ","

    def __init__(self, line: int, col: int):
        self.line: int = line
        self.col: int = col

    def __str__(self):
        return f"(Ln {self.line}, Col {self.col})"

    @staticmethod
    def from_str(pos: str) -> "Position":
        line, col = pos.split(Position.separator)
        return Position(int(line) - 1, int(col) - 1)


class PatchCommand:
    separator = "|"

    def __init__(self, filepath: str, start: Position, end: Position, content: str):
        self.filepath: str = filepath
        self.start: Position = start
        self.end: Position = end
        self.content: str = content

    def read_lines(self) -> list[str]:
        with open(self.filepath, "r") as f:
            lines = f.readlines()
        return lines

    def write_lines(self, lines: list[str]) -> int:
        with open(self.filepath, "w") as f:
            f.writelines(lines)
        return sum([len(line) for line in lines])

    @verify
    def execute(self) -> Tuple[int, int]:
        lines = self.read_lines()
        before = sum([len(line) for line in lines])

        lines[self.start.line] = (
            lines[self.start.line][: self.start.col]
            + self.content
            + lines[self.end.line][self.end.col :]
        )
        lines = lines[: self.start.line + 1] + lines[self.end.line + 1 :]

        after = self.write_lines(lines)

        written = len(self.content)
        deleted = before - after + written

        return written, deleted

    @staticmethod
    def from_str(command: str) -> "PatchCommand":
        match = re.search(
            r"(.*)\|([0-9]*),([0-9]*)\|([0-9]*),([0-9]*)(\||\n)(.*)",
            command,
            re.DOTALL,
        )
        filepath = match.group(1)
        start_line = match.group(2)
        start_col = match.group(3)
        end_line = match.group(4)
        end_col = match.group(5)
        content = match.group(7)
        return PatchCommand(
            filepath,
            Position.from_str(f"{start_line},{start_col}"),
            Position.from_str(f"{end_line},{end_col}"),
            content,
        )


class CodePatcher:
    separator = "\n---~~~+++===+++~~~---\n"

    @staticmethod
    def sort_commands(commands: list[PatchCommand]) -> list[PatchCommand]:
        return sorted(commands, key=lambda c: c.start.line, reverse=True)

    @staticmethod
    def patch(bulk_command: str) -> Tuple[int, int]:
        commands = [
            PatchCommand.from_str(command)
            for command in bulk_command.split(CodePatcher.separator)
            if command != ""
        ]
        commands = CodePatcher.sort_commands(commands)

        written, deleted = 0, 0
        for command in commands:
            if command:
                w, d = command.execute()
                written += w
                deleted += d
        return written, deleted


# if __name__ == "__main__":
#     commands = """test.py|2,1|2,1|from bs4 import BeautifulSoup

# ---~~~+++===+++~~~---
# test.py|5,5|5,33|html = requests.get(url).text
#     soup = BeautifulSoup(html, "html.parser")
#     news_results = soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd")
# ---~~~+++===+++~~~---
# test.py|7,5|9,13|news_titles = []
#     for result in news_results:
#         news_titles
# ---~~~+++===+++~~~---
# test.py|11,16|11,16|_titles
# """

#     example = """import requests

# def crawl_news(keyword):
#     url = f"https://www.google.com/search?q={keyword}+news"
#     response = requests.get(url)

#     news = []
#     for result in response:
#         news.append(result.text)

#     return news
# """
#     testfile = "test.py"
#     with open(testfile, "w") as f:
#         f.write(example)

#     patcher = CodePatcher()
#     written, deleted = patcher.patch(commands)
#     print(f"written: {written}, deleted: {deleted}")

####################### => EDITOR/PATCH.PY






###################### EDITOR// INIT.PY


class CodeEditor(BaseToolSet):
    @tool(
        name="CodeEditor.READ",
        description="Read and understand code. "
        "Input should be filename and line number group. ex. test.py|1-10 "
        "and the output will be code. ",
    )
    def read(self, inputs: str) -> str:
        try:
            output = CodeReader.read(inputs)
        except Exception as e:
            output = str(e)

        logger.debug(
            f"\nProcessed CodeEditor.READ, Input Commands: {inputs} "
            f"Output Answer: {output}"
        )
        return output

    @tool(
        name="CodeEditor.SUMMARY",
        description="Summary code. "
        "Read the code structured into a tree. "
        "If you set specific line, it will show the code from the specific line. "
        "Input should be filename, depth, and specific line if you want. ex. test.py|2 or test.py|3|print('hello world') "
        "and the output will be list of (line number: code). ",
    )
    def summary(self, inputs: str) -> str:
        try:
            output = CodeReader.summary(inputs)
        except Exception as e:
            output = str(e)

        logger.debug(
            f"\nProcessed CodeEditor.SUMMARY, Input Commands: {inputs} "
            f"Output Answer: {output}"
        )
        return output

    @tool(
        name="CodeEditor.APPEND",
        description="Append code to the existing file. "
        "If the code is completed, use the Terminal tool to execute it, if not, append the code through the this tool. "
        "Input should be filename and code to append. "
        "Input code must be the code that should be appended, NOT whole code. "
        "ex. test.py\nprint('hello world')\n "
        "and the output will be last 3 lines.",
    )
    def append(self, inputs: str) -> str:
        try:
            code = CodeWriter.append(inputs)
            output = "Last 3 line was:\n" + "\n".join(code.split("\n")[-3:])
        except Exception as e:
            output = str(e)

        logger.debug(
            f"\nProcessed CodeEditor.APPEND, Input: {inputs} "
            f"Output Answer: {output}"
        )
        return output

    @tool(
        name="CodeEditor.WRITE",
        description="Write code to create a new tool. "
        "If the code is completed, use the Terminal tool to execute it, if not, append the code through the CodeEditor.APPEND tool. "
        "Input should be formatted like: "
        "<filename>\n<code>\n\n"
        "Here is an example: "
        "test.py\nmessage = 'hello world'\nprint(message)\n"
        "\n"
        "The output will be last 3 lines you wrote.",
    )
    def write(self, inputs: str) -> str:
        try:
            code = CodeWriter.write(inputs.lstrip())
            output = "Last 3 line was:\n" + "\n".join(code.split("\n")[-3:])
        except Exception as e:
            output = str(e)

        logger.debug(
            f"\nProcessed CodeEditor.WRITE, Input: {inputs} " f"Output Answer: {output}"
        )
        return output

    # @tool(
    #     name="CodeEditor.PATCH",
    #     description="Patch the code to correct the error if an error occurs or to improve it. "
    #     "Input is a list of patches. The patch is separated by {seperator}. ".format(
    #         seperator=CodePatcher.separator.replace("\n", "\\n")
    #     )
    #     + "Each patch has to be formatted like below.\n"
    #     "<filepath>|<start_line>,<start_col>|<end_line>,<end_col>|<new_code>"
    #     "Here is an example. If the original code is:\n"
    #     "print('hello world')\n"
    #     "and you want to change it to:\n"
    #     "print('hi corca')\n"
    #     "then the patch should be:\n"
    #     "test.py|1,8|1,19|hi corca\n"
    #     "Code between start and end will be replaced with new_code. "
    #     "The output will be written/deleted bytes or error message. ",
    # )
    def patch(self, patches: str) -> str:
        try:
            w, d = CodePatcher.patch(patches)
            output = f"successfully wrote {w}, deleted {d}"
        except Exception as e:
            output = str(e)

        logger.debug(
            f"\nProcessed CodeEditor.PATCH, Input Patch: {patches} "
            f"Output Answer: {output}"
        )
        return output

    @tool(
        name="CodeEditor.DELETE",
        description="Delete code in file for a new start. "
        "Input should be filename."
        "ex. test.py "
        "Output will be success or error message.",
    )
    def delete(self, inputs: str, filepath: str) -> str:
        try:
            with open(filepath, "w") as f:
                f.write("")
            output = "success"
        except Exception as e:
            output = str(e)

        logger.debug(
            f"\nProcessed CodeEditor.DELETE, Input filename: {inputs} "
            f"Output Answer: {output}"
        )
        return output
    
###################### EDITOR// INIT.PY END






########################### MODELS
import uuid

import numpy as np
import torch
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline,
)
from PIL import Image
from transformers import (
    BlipForQuestionAnswering,
    BlipProcessor,
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
)


from swarms.utils.main import get_new_image_name


class MaskFormer(BaseToolSet):
    def __init__(self, device):
        print("Initializing MaskFormer to %s" % device)
        self.device = device
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        ).to(device)

    def inference(self, image_path, text):
        threshold = 0.5
        min_area = 0.02
        padding = 20
        original_image = Image.open(image_path)
        image = original_image.resize((512, 512))
        inputs = self.processor(
            text=text, images=image, padding="max_length", return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        mask = torch.sigmoid(outputs[0]).squeeze().cpu().numpy() > threshold
        area_ratio = len(np.argwhere(mask)) / (mask.shape[0] * mask.shape[1])
        if area_ratio < min_area:
            return None
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(
                slice(max(0, i - padding), i + padding + 1) for i in idx
            )
            mask_array[padded_slice] = True
        visual_mask = (mask_array * 255).astype(np.uint8)
        image_mask = Image.fromarray(visual_mask)
        return image_mask.resize(original_image.size)


class ImageEditing(BaseToolSet):
    def __init__(self, device):
        print("Initializing ImageEditing to %s" % device)
        self.device = device
        self.mask_former = MaskFormer(device=self.device)
        self.revision = "fp16" if "cuda" in device else None
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            revision=self.revision,
            torch_dtype=self.torch_dtype,
        ).to(device)

    @tool(
        name="Remove Something From The Photo",
        description="useful when you want to remove and object or something from the photo "
        "from its description or location. "
        "The input to this tool should be a comma separated string of two, "
        "representing the image_path and the object need to be removed. ",
    )
    def inference_remove(self, inputs):
        image_path, to_be_removed_txt = inputs.split(",")
        return self.inference_replace(f"{image_path},{to_be_removed_txt},background")

    @tool(
        name="Replace Something From The Photo",
        description="useful when you want to replace an object from the object description or "
        "location with another object from its description. "
        "The input to this tool should be a comma separated string of three, "
        "representing the image_path, the object to be replaced, the object to be replaced with ",
    )
    def inference_replace(self, inputs):
        image_path, to_be_replaced_txt, replace_with_txt = inputs.split(",")
        original_image = Image.open(image_path)
        original_size = original_image.size
        mask_image = self.mask_former.inference(image_path, to_be_replaced_txt)
        updated_image = self.inpaint(
            prompt=replace_with_txt,
            image=original_image.resize((512, 512)),
            mask_image=mask_image.resize((512, 512)),
        ).images[0]
        updated_image_path = get_new_image_name(
            image_path, func_name="replace-something"
        )
        updated_image = updated_image.resize(original_size)
        updated_image.save(updated_image_path)

        logger.debug(
            f"\nProcessed ImageEditing, Input Image: {image_path}, Replace {to_be_replaced_txt} to {replace_with_txt}, "
            f"Output Image: {updated_image_path}"
        )

        return updated_image_path


class InstructPix2Pix(BaseToolSet):
    def __init__(self, device):
        print("Initializing InstructPix2Pix to %s" % device)
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        ).to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

    @tool(
        name="Instruct Image Using Text",
        description="useful when you want to the style of the image to be like the text. "
        "like: make it look like a painting. or make it like a robot. "
        "The input to this tool should be a comma separated string of two, "
        "representing the image_path and the text. ",
    )
    def inference(self, inputs):
        """Change style of image."""
        logger.debug("===> Starting InstructPix2Pix Inference")
        image_path, text = inputs.split(",")[0], ",".join(inputs.split(",")[1:])
        original_image = Image.open(image_path)
        image = self.pipe(
            text, image=original_image, num_inference_steps=40, image_guidance_scale=1.2
        ).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="pix2pix")
        image.save(updated_image_path)

        logger.debug(
            f"\nProcessed InstructPix2Pix, Input Image: {image_path}, Instruct Text: {text}, "
            f"Output Image: {updated_image_path}"
        )

        return updated_image_path


class Text2Image(BaseToolSet):
    def __init__(self, device):
        print("Initializing Text2Image to %s" % device)
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=self.torch_dtype
        )
        self.pipe.to(device)
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        )

    @tool(
        name="Generate Image From User Input Text",
        description="useful when you want to generate an image from a user input text and save it to a file. "
        "like: generate an image of an object or something, or generate an image that includes some objects. "
        "The input to this tool should be a string, representing the text used to generate image. ",
    )
    def inference(self, text):
        image_filename = os.path.join("image", str(uuid.uuid4())[0:8] + ".png")
        prompt = text + ", " + self.a_prompt
        image = self.pipe(prompt, negative_prompt=self.n_prompt).images[0]
        image.save(image_filename)

        logger.debug(
            f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}"
        )

        return image_filename


class VisualQuestionAnswering(BaseToolSet):
    def __init__(self, device):
        print("Initializing VisualQuestionAnswering to %s" % device)
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base", torch_dtype=self.torch_dtype
        ).to(self.device)

    @tool(
        name="Answer Question About The Image",
        description="useful when you need an answer for a question based on an image. "
        "like: what is the background color of the last image, how many cats in this figure, what is in this figure. "
        "The input to this tool should be a comma separated string of two, representing the image_path and the question",
    )
    def inference(self, inputs):
        image_path, question = inputs.split(",")
        raw_image = Image.open(image_path).convert("RGB")
        inputs = self.processor(raw_image, question, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)

        logger.debug(
            f"\nProcessed VisualQuestionAnswering, Input Image: {image_path}, Input Question: {question}, "
            f"Output Answer: {answer}"
        )

        return answer
    

#========================> handlers/image
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

# from core.prompts.file import IMAGE_PROMPT
# from swarms.agents.prompts import IMAGE_PROMPT
from swarms.agents.prompts.prompts import IMAGE_PROMPT

from swarms.utils.main import BaseHandler

class ImageCaptioning(BaseHandler):
    def __init__(self, device):
        print("Initializing ImageCaptioning to %s" % device)
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype
        ).to(self.device)

    def handle(self, filename: str):
        img = Image.open(filename)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        img = img.resize((width_new, height_new))
        img = img.convert("RGB")
        img.save(filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")

        inputs = self.processor(Image.open(filename), return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs)
        description = self.processor.decode(out[0], skip_special_tokens=True)
        print(
            f"\nProcessed ImageCaptioning, Input Image: {filename}, Output Text: {description}"
        )

        return IMAGE_PROMPT.format(filename=filename, description=description)
    






#segment anything:

########################### MODELS


# #########==========================> 
# from selenium import webdriver
# from langchain.tools import BaseTool

# class BrowserActionTool(BaseTool):
#     name = "browser_action"
#     description = "Perform a browser action."

#     prompt = """
    
#     Sure, here are few-shot prompts for each of the browser tools:

#     1. **Go To URL Tool**
#     Prompt: "Navigate to the OpenAI homepage."
#     Command: `{ "action_type": "go_to", "url": "https://www.openai.com" }`

#     2. **Form Submission Tool**
#     Prompt: "On the page 'https://www.formexample.com', find the form with the id 'login', set the 'username' field to 'testuser', and the 'password' field to 'testpassword', then submit the form."
#     Command: `{ "action_type": "submit_form", "form_id": "login", "form_values": { "username": "testuser", "password": "testpassword" } }`

#     3. **Click Link Tool**
#     Prompt: "On the current page, find the link with the text 'About Us' and click it."
#     Command: `{ "action_type": "click_link", "link_text": "About Us" }`

#     4. **Enter Text Tool**
#     Prompt: "On the page 'https://www.textentryexample.com', find the text area with the id 'message' and enter the text 'Hello World'."
#     Command: `{ "action_type": "enter_text", "text_area_id": "message", "text": "Hello World" }`

#     5. **Button Click Tool**
#     Prompt: "On the current page, find the button with the id 'submit' and click it."
#     Command: `{ "action_type": "click_button", "button_id": "submit" }`

#     6. **Select Option Tool**
#     Prompt: "On the page 'https://www.selectoptionexample.com', find the select dropdown with the id 'country' and select the option 'United States'."
#     Command: `{ "action_type": "select_option", "select_id": "country", "option": "United States" }`

#     7. **Hover Tool**
#     Prompt: "On the current page, find the element with the id 'menu' and hover over it."
#     Command: `{ "action_type": "hover", "element_id": "menu" }`

#     8. **Scroll Tool**
#     Prompt: "On the current page, scroll down to the element with the id 'footer'."
#     Command: `{ "action_type": "scroll", "element_id": "footer" }`

#     9. **Screenshot Tool**
#     Prompt: "On the current page, take a screenshot."
#     Command: `{ "action_type": "screenshot" }`

#     10. **Back Navigation Tool**
#     Prompt: "Navigate back to the previous page."
#     Command: `{ "action_type": "back" }`

    
#     """

#     def _run(self, action_type: str, action_details: dict) -> str:
#         """Perform a browser action based on action_type and action_details."""

#         try:
#             driver = webdriver.Firefox()

#             if action_type == 'Open Browser':
#                 pass  # Browser is already opened
#             elif action_type == 'Close Browser':
#                 driver.quit()
#             elif action_type == 'Navigate To URL':
#                 driver.get(action_details['url'])
#             elif action_type == 'Fill Form':
#                 for field_name, field_value in action_details['fields'].items():
#                     element = driver.find_element_by_name(field_name)
#                     element.send_keys(field_value)
#             elif action_type == 'Submit Form':
#                 element = driver.find_element_by_name(action_details['form_name'])
#                 element.submit()
#             elif action_type == 'Click Button':
#                 element = driver.find_element_by_name(action_details['button_name'])
#                 element.click()
#             elif action_type == 'Scroll Down':
#                 driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#             elif action_type == 'Scroll Up':
#                 driver.execute_script("window.scrollTo(0, 0);")
#             elif action_type == 'Go Back':
#                 driver.back()
#             elif action_type == 'Go Forward':
#                 driver.forward()
#             elif action_type == 'Refresh':
#                 driver.refresh()
#             elif action_type == 'Execute Javascript':
#                 driver.execute_script(action_details['script'])
#             elif action_type == 'Switch Tab':
#                 driver.switch_to.window(driver.window_handles[action_details['tab_index']])
#             elif action_type == 'Close Tab':
#                 driver.close()
#             else:
#                 return f"Error: Unknown action type {action_type}."

#             return f"Action {action_type} completed successfully."
#         except Exception as e:
#             return f"Error: {e}"


#--------------------------------------> END





#--------------------------------------> AUTO GPT TOOLS

# General 
import os
import pandas as pd

from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.docstore.document import Document
import asyncio

# Tools
from contextlib import contextmanager
from typing import Optional
from langchain.agents import tool

ROOT_DIR = "./data/"

from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pydantic import Field
from langchain.chains.qa_with_sources.loading import BaseCombineDocumentsChain



@contextmanager
def pushd(new_dir):
    """Context manager for changing the current working directory."""
    prev_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)

@tool
def process_csv(
    llm, csv_file_path: str, instructions: str, output_path: Optional[str] = None
) -> str:
    """Process a CSV by with pandas in a limited REPL.\
 Only use this after writing data to disk as a csv file.\
 Any figures must be saved to disk to be viewed by the human.\
 Instructions should be written in natural language, not code. Assume the dataframe is already loaded."""
    with pushd(ROOT_DIR):
        try:
            df = pd.read_csv(csv_file_path)
        except Exception as e:
            return f"Error: {e}"
        agent = create_pandas_dataframe_agent(llm, df, max_iterations=30, verbose=False)
        if output_path is not None:
            instructions += f" Save output to disk at {output_path}"
        try:
            result = agent.run(instructions)
            return result
        except Exception as e:
            return f"Error: {e}"
        

async def async_load_playwright(url: str) -> str:
    """Load the specified URLs using Playwright and parse using BeautifulSoup."""
    from bs4 import BeautifulSoup
    from playwright.async_api import async_playwright

    results = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url)

            page_source = await page.content()
            soup = BeautifulSoup(page_source, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            results = "\n".join(chunk for chunk in chunks if chunk)
        except Exception as e:
            results = f"Error: {e}"
        await browser.close()
    return results

def run_async(coro):
    event_loop = asyncio.get_event_loop()
    return event_loop.run_until_complete(coro)

@tool
def browse_web_page(url: str) -> str:
    """Verbose way to scrape a whole webpage. Likely to cause issues parsing."""
    return run_async(async_load_playwright(url))


def _get_text_splitter():
    return RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 500,
        chunk_overlap  = 20,
        length_function = len,
    )


class WebpageQATool(BaseTool):
    name = "query_webpage"
    description = "Browse a webpage and retrieve the information relevant to the question."
    text_splitter: RecursiveCharacterTextSplitter = Field(default_factory=_get_text_splitter)
    qa_chain: BaseCombineDocumentsChain
    
    def _run(self, url: str, question: str) -> str:
        """Useful for browsing websites and scraping the text information."""
        result = browse_web_page.run(url)
        docs = [Document(page_content=result, metadata={"source": url})]
        web_docs = self.text_splitter.split_documents(docs)
        results = []
        # TODO: Handle this with a MapReduceChain
        for i in range(0, len(web_docs), 4):
            input_docs = web_docs[i:i+4]
            window_result = self.qa_chain({"input_documents": input_docs, "question": question}, return_only_outputs=True)
            results.append(f"Response from window {i} - {window_result}")
        results_docs = [Document(page_content="\n".join(results), metadata={"source": url})]
        return self.qa_chain({"input_documents": results_docs, "question": question}, return_only_outputs=True)
    
    async def _arun(self, url: str, question: str) -> str:
        raise NotImplementedError


# query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))

# !pip install duckduckgo_search
web_search = DuckDuckGoSearchRun()








######################################################## zapier


# get from https://platform.openai.com/
# os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")

# # get from https://nla.zapier.com/demo/provider/debug (under User Information, after logging in):
# os.environ["ZAPIER_NLA_API_KEY"] = os.environ.get("ZAPIER_NLA_API_KEY", "")


# from langchain.agents.agent_toolkits import ZapierToolkit
# from langchain.agents import AgentType
# from langchain.utilities.zapier import ZapierNLAWrapper


# zapier = ZapierNLAWrapper()
# zapier_toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
# # agent = initialize_agent(
# #     toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
# # )


######################################################## zapier end




######################################################## youtube search
# from langchain.tools import YouTubeSearchTool

# youtube_tool = YouTubeSearchTool()

# #tool.run("lex friedman")

######################################################## youtube search end




######################################################## wolfram beginning

# import os

# os.environ["WOLFRAM_ALPHA_APPID"] = ""

# from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

# wolfram_tool = WolframAlphaAPIWrapper()
# #wolfram.run("What is 2x+5 = -3x + 7?")

######################################################## wolfram end


######################################################## Wikipedia beginning
# from langchain.utilities import WikipediaAPIWrapper

# wikipedia_tool = WikipediaAPIWrapper()

# #wikipedia.run("HUNTER X HUNTER")
######################################################## Wikipedia beginning



######################################################## search tools beginning

# google_serpe_tools = load_tools(["google-serper"])

######################################################## search tools end



######################################################## requests

# from langchain.agents import load_tools

# requests_tools = load_tools(["requests_all"])
# # requests_tools

# requests_tools[0].requests_wrapper


# from langchain.utilities import TextRequestsWrapper


# requests = TextRequestsWrapper()

# requests.get("https://www.google.com")

######################################################## requests


######################################################## pubmed
# from langchain.tools import PubmedQueryRun

# pubmed_tool = PubmedQueryRun()

# pubmed_tool.run("chatgpt")


######################################################## pubmed emd



######################################################## IFTTT WebHooks

# from langchain.tools.ifttt import IFTTTWebhook


# import os

# key = os.environ["IFTTTKey"]
# url = f"https://maker.ifttt.com/trigger/spotify/json/with/key/{key}"
# IFFT_tool = IFTTTWebhook(
#     name="Spotify", description="Add a song to spotify playlist", url=url
# )


######################################################## IFTTT WebHooks end



######################################################## huggingface
# from langchain.agents import load_huggingface_tool

# hf_tool = load_huggingface_tool("lysandre/hf-model-downloads")

# print(f"{tool.name}: {tool.description}")


######################################################## huggingface end


######################################################## graphql

# from langchain import OpenAI
# from langchain.agents import load_tools, initialize_agent, AgentType
# from langchain.utilities import GraphQLAPIWrapper

# llm = OpenAI(temperature=0)

# graphql_tool = load_tools(
#     ["graphql"],
#     graphql_endpoint="https://swapi-graphql.netlify.app/.netlify/functions/index"
# )

# agent = initialize_agent(
#     tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
# )


######################################################## graphql end



######################################################## graphql 
# from langchain.agents import initialize_agent
# from langchain.llms import OpenAI
# from gradio_tools.tools import (
#     StableDiffusionTool,
#     ImageCaptioningTool,
#     StableDiffusionPromptGeneratorTool,
#     TextToVideoTool,
# )

# from langchain.memory import ConversationBufferMemory

# hf_model_tools = [
#     StableDiffusionTool().langchain,
#     ImageCaptioningTool().langchain,
#     StableDiffusionPromptGeneratorTool().langchain,
#     TextToVideoTool().langchain,
# ]


######################## ######################################################## graphql end 






######################## ######################################################## file system

from langchain.agents.agent_toolkits import FileManagementToolkit
from tempfile import TemporaryDirectory

# We'll make a temporary directory to avoid clutter
working_directory = TemporaryDirectory()

toolkit = FileManagementToolkit(
    root_dir=str(working_directory.name)
)  # If you don't provide a root_dir, operations will default to the current working directory
toolkit.get_tools()

file_management_tools = FileManagementToolkit(
    root_dir=str(working_directory.name),
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools()

read_tool, write_tool, list_tool = file_management_tools
write_tool.run({"file_path": "example.txt", "text": "Hello World!"})

# List files in the working directory
list_tool.run({})


######################### BRAVE

# from langchain.tools import BraveSearch

# brave_api_key = os.environ["BRAVE_API_KEY"]

# brave_tool = BraveSearch.from_api_key(api_key=brave_api_key, search_kwargs={"count": 3})



######################### BRAVE END



######################### ARXVIV


# from langchain.chat_models import ChatOpenAI
# from langchain.agents import load_tools, initialize_agent, AgentType


# arxviv_tool = load_tools(
#     ["arxiv"],
# )

# ############

# from langchain.utilities import ArxivAPIWrapper

# arxiv_tool = ArxivAPIWrapper()



# ################################# GMAIL TOOKKIT 
# from langchain.agents.agent_toolkits import GmailToolkit

# gmail_toolkit = GmailToolkit()


# from langchain.tools.gmail.utils import build_resource_service, get_gmail_credentials

# # Can review scopes here https://developers.google.com/gmail/api/auth/scopes
# # For instance, readonly scope is 'https://www.googleapis.com/auth/gmail.readonly'
# credentials = get_gmail_credentials(
#     token_file="token.json",
#     scopes=["https://mail.google.com/"],
#     client_secrets_file="credentials.json",
# )

# api_resource = build_resource_service(credentials=credentials)
# gmail_toolkit_2 = GmailToolkit(api_resource=api_resource)

# gmail_tools = toolkit.get_tools()

# from langchain import OpenAI
# from langchain.agents import initialize_agent, AgentType


# agent = initialize_agent(
#     tools=toolkit.get_tools(),
#     llm=llm,
#     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
# )




################################# GMAIL TOOKKIT  JSON AGENT
# import os
# import yaml

# from langchain.agents import create_json_agent, AgentExecutor
# from langchain.agents.agent_toolkits import JsonToolkit
# from langchain.chains import LLMChain
# from langchain.llms.openai import OpenAI
# from langchain.requests import TextRequestsWrapper
# from langchain.tools.json.tool import JsonSpec

# with open("openai_openapi.yml") as f:
#     data = yaml.load(f, Loader=yaml.FullLoader)
# json_spec = JsonSpec(dict_=data, max_value_length=4000)
# json_toolkit = JsonToolkit(spec=json_spec)

# json_agent_executor = create_json_agent(
#     llm=OpenAI(temperature=0), toolkit=json_toolkit, verbose=True
# )

# json_agent_executor.run(
#     "What are the required parameters in the request body to the /completions endpoint?"
# )

# ################################# OFFICE 365 TOOLKIT

# from langchain.agents.agent_toolkits import O365Toolkit

# threesixfive_toolkit = O365Toolkit()

# threesixfive_toolkit = toolkit.get_tools()


################################# OFFICE 365 TOOLKIT END


# import os, yaml

# wget https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
# mv openapi.yaml openai_openapi.yaml
# wget https://www.klarna.com/us/shopping/public/openai/v0/api-docs
# mv api-docs klarna_openapi.yaml
# wget https://raw.githubusercontent.com/APIs-guru/openapi-directory/main/APIs/spotify.com/1.0.0/openapi.yaml
# mv openapi.yaml spotify_openapi.yaml

# from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec

# with open("openai_openapi.yaml") as f:
#     raw_openai_api_spec = yaml.load(f, Loader=yaml.Loader)
# openai_api_spec = reduce_openapi_spec(raw_openai_api_spec)

# with open("klarna_openapi.yaml") as f:
#     raw_klarna_api_spec = yaml.load(f, Loader=yaml.Loader)
# klarna_api_spec = reduce_openapi_spec(raw_klarna_api_spec)

# with open("spotify_openapi.yaml") as f:
#     raw_spotify_api_spec = yaml.load(f, Loader=yaml.Loader)
# spotify_api_spec = reduce_openapi_spec(raw_spotify_api_spec)

# import spotipy.util as util
# from langchain.requests import RequestsWrapper


# def construct_spotify_auth_headers(raw_spec: dict):
#     scopes = list(
#         raw_spec["components"]["securitySchemes"]["oauth_2_0"]["flows"][
#             "authorizationCode"
#         ]["scopes"].keys()
#     )
#     access_token = util.prompt_for_user_token(scope=",".join(scopes))
#     return {"Authorization": f"Bearer {access_token}"}


# # Get API credentials.
# headers = construct_spotify_auth_headers(raw_spotify_api_spec)
# requests_wrapper = RequestsWrapper(headers=headers)


# endpoints = [
#     (route, operation)
#     for route, operations in raw_spotify_api_spec["paths"].items()
#     for operation in operations
#     if operation in ["get", "post"]
# ]

# len(endpoints)

# import tiktoken

# enc = tiktoken.encoding_for_model("text-davinci-003")


# def count_tokens(s):
#     return len(enc.encode(s))


# count_tokens(yaml.dump(raw_spotify_api_spec))

# from langchain.llms.openai import OpenAI
# from langchain.agents.agent_toolkits.openapi import planner

# llm = OpenAI(model_name="gpt-4", temperature=0.0, openai_api_key=openai_api_key)

# spotify_agent = planner.create_openapi_agent(spotify_api_spec, requests_wrapper, llm)
# user_query = (
#     "make me a playlist with the first song from kind of blue. call it machine blues."
# )
# spotify_agent.run(user_query)


# from langchain.agents import create_openapi_agent
# from langchain.agents.agent_toolkits import OpenAPIToolkit
# from langchain.llms.openai import OpenAI
# from langchain.requests import TextRequestsWrapper
# from langchain.tools.json.tool import JsonSpec

# with open("openai_openapi.yaml") as f:
#     data = yaml.load(f, Loader=yaml.FullLoader)
# json_spec = JsonSpec(dict_=data, max_value_length=4000)


# openapi_toolkit = OpenAPIToolkit.from_llm(
#     OpenAI(temperature=0), json_spec, openai_requests_wrapper, verbose=True
# )
# openapi_agent_executor = create_openapi_agent(
#     llm=OpenAI(temperature=0), toolkit=openapi_toolkit, verbose=True
# )


############################################ Natural Language APIs start

# from typing import List, Optional
# from langchain.chains import LLMChain
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.requests import Requests
# from langchain.tools import APIOperation, OpenAPISpec
# from langchain.agents import AgentType, Tool, initialize_agent
# from langchain.agents.agent_toolkits import NLAToolkit

# # Select the LLM to use. Here, we use text-davinci-003
# llm = OpenAI(
#     temperature=0, max_tokens=700, openai_api_key=openai_api_key
# )  # You can swap between different core LLM's here.

# speak_toolkit = NLAToolkit.from_llm_and_url(llm, "https://api.speak.com/openapi.yaml")
# klarna_toolkit = NLAToolkit.from_llm_and_url(
#     llm, "https://www.klarna.com/us/shopping/public/openai/v0/api-docs/"
# )

# # Slightly tweak the instructions from the default agent
# openapi_format_instructions = """Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: what to instruct the AI Action representative.
# Observation: The Agent's response
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer. User can't see any of my observations, API responses, links, or tools.
# Final Answer: the final answer to the original input question with the right amount of detail

# When responding with your Final Answer, remember that the person you are responding to CANNOT see any of your Thought/Action/Action Input/Observations, so if there is any relevant information there you need to include it explicitly in your response."""

# natural_language_tools = speak_toolkit.get_tools() + klarna_toolkit.get_tools()
# mrkl = initialize_agent(
#     natural_language_tools,
#     llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     agent_kwargs={"format_instructions": openapi_format_instructions},
# )

# mrkl.run(
#     "I have an end of year party for my Italian class and have to buy some Italian clothes for it"
# )

# spoonacular_api = os.environ["SPOONACULAR_KEY"]
# spoonacular_api_key = spoonacular_api

# requests = Requests(headers={"x-api-key": spoonacular_api_key})
# spoonacular_toolkit = NLAToolkit.from_llm_and_url(
#     llm,
#     "https://spoonacular.com/application/frontend/downloads/spoonacular-openapi-3.json",
#     requests=requests,
#     max_text_length=1800,  # If you want to truncate the response text
# )

# natural_language_api_tools = (
#     speak_toolkit.get_tools()
#     + klarna_toolkit.get_tools()
#     + spoonacular_toolkit.get_tools()[:30]
# )
# print(f"{len(natural_language_api_tools)} tools loaded.")


# natural_language_api_tools[1].run(
#     "Tell the LangChain audience to 'enjoy the meal' in Italian, please!"
# )

############################################ Natural Language APIs start END









############################################ python tool
# from langchain.agents.agent_toolkits import create_python_agent
# from langchain.tools.python.tool import PythonREPLTool
# from langchain.python import PythonREPL
# from langchain.llms.openai import OpenAI
# from langchain.agents.agent_types import AgentType
# from langchain.chat_models import ChatOpenAI


# #test
# # PythonREPLTool()
# python_repl_tool = PythonREPLTool()
############################################ python tool


############### VECTOR STORE CHROMA, MAKE OCEAN

# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import CharacterTextSplitter
# from langchain import OpenAI, VectorDBQA

# llm = OpenAI(temperature=0, openai_api_key=openai_api_key)


# from langchain.document_loaders import TextLoader

# loader = TextLoader("../../../state_of_the_union.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings()
# state_of_union_store = Chroma.from_documents(
#     texts, embeddings, collection_name="state-of-union"
# )

# from langchain.document_loaders import WebBaseLoader

# loader = WebBaseLoader("https://beta.ruff.rs/docs/faq/")
# docs = loader.load()
# ruff_texts = text_splitter.split_documents(docs)
# ruff_store = Chroma.from_documents(ruff_texts, embeddings, collection_name="ruff")


# ############ Initialize Toolkit and Agent
# from langchain.agents.agent_toolkits import (
#     create_vectorstore_agent,
#     VectorStoreToolkit,
#     VectorStoreInfo,
# )

# vectorstore_info = VectorStoreInfo(
#     name="state_of_union_address",
#     description="the most recent state of the Union adress",
#     vectorstore=state_of_union_store,
# )
# vectorstore_toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
# agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)




######################### Multiple Vectorstores
#We can also easily use this initialize an agent with multiple vectorstores and use the agent to route between them. To do this. This agent is optimized for routing, so it is a different toolkit and initializer.


# from langchain.agents.agent_toolkits import (
#     create_vectorstore_router_agent,
#     VectorStoreRouterToolkit,
#     VectorStoreInfo,
# )

# ruff_vectorstore_info = VectorStoreInfo(
#     name="ruff",
#     description="Information about the Ruff python linting library",
#     vectorstore=ruff_store,
# )
# router_toolkit = VectorStoreRouterToolkit(
#     vectorstores=[vectorstore_info, ruff_vectorstore_info], llm=llm
# )
# #





############################################### ===========================> Whisperx speech to text
# import os
# from pydantic import BaseModel, Field
# from pydub import AudioSegment
# from pytube import YouTube
# import whisperx
# from langchain.tools import tool


# hf_api_key = os.environ["HF_API_KEY"]
# # define a custom input schema for the youtube url
# class YouTubeVideoInput(BaseModel):
#     video_url: str = Field(description="YouTube Video URL to transcribe")


# def download_youtube_video(video_url, audio_format='mp3'):
#     audio_file = f'video.{audio_format}'
    
#     # Download video
#     yt = YouTube(video_url)
#     yt_stream = yt.streams.filter(only_audio=True).first()
#     yt_stream.download(filename='video.mp4')

#     # Convert video to audio
#     video = AudioSegment.from_file("video.mp4", format="mp4")
#     video.export(audio_file, format=audio_format)
#     os.remove("video.mp4")
    
#     return audio_file


# @tool("transcribe_youtube_video", args_schema=YouTubeVideoInput, return_direct=True)
# def transcribe_youtube_video(video_url: str) -> str:
#     """Transcribes a YouTube video."""
#     audio_file = download_youtube_video(video_url)
    
#     device = "cuda"
#     batch_size = 16
#     compute_type = "float16"

#     # 1. Transcribe with original Whisper (batched)
#     model = whisperx.load_model("large-v2", device, compute_type=compute_type)
#     audio = whisperx.load_audio(audio_file)
#     result = model.transcribe(audio, batch_size=batch_size)

#     # 2. Align Whisper output
#     model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
#     result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

#     # 3. Assign speaker labels

#     diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_api_key, device=device)
#     diarize_segments = diarize_model(audio_file)
    
#     try:
#       segments = result["segments"]
#       transcription = " ".join(segment['text'] for segment in segments)
#       return transcription
#     except KeyError:
#       print("The key 'segments' is not found in the result.")



# ################################################### BASE WHISPER TOOL
# from typing import Optional, Type
# from pydantic import BaseModel, Field
# from langchain.tools import BaseTool
# from langchain.callbacks.manager import (
#     AsyncCallbackManagerForToolRun,
#     CallbackManagerForToolRun,
# )
# import requests
# import whisperx

# class AudioInput(BaseModel):
#     audio_file: str = Field(description="Path to audio file")


# class TranscribeAudioTool(BaseTool):
#     name = "transcribe_audio"
#     description = "Transcribes an audio file using WhisperX"
#     args_schema: Type[AudioInput] = AudioInput

#     def _run(
#         self,
#         audio_file: str,
#         device: str = "cuda",
#         batch_size: int = 16,
#         compute_type: str = "float16",
#         run_manager: Optional[CallbackManagerForToolRun] = None,
#     ) -> str:
#         """Use the tool."""
#         model = whisperx.load_model("large-v2", device, compute_type=compute_type)
#         audio = whisperx.load_audio(audio_file)
#         result = model.transcribe(audio, batch_size=batch_size)
        
#         model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
#         result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

#         diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_api_key, device=device)
#         diarize_segments = diarize_model(audio_file)
        
#         try:
#             segments = result["segments"]
#             transcription = " ".join(segment['text'] for segment in segments)
#             return transcription
#         except KeyError:
#             print("The key 'segments' is not found in the result.")

#     async def _arun(
#         self,
#         audio_file: str,
#         device: str = "cuda",
#         batch_size: int = 16,
#         compute_type: str = "float16",
#         run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
#     ) -> str:
#         """Use the tool asynchronously."""
#         raise NotImplementedError("transcribe_audio does not support async")
###########=========================>

# #======> Calculator
# from langchain import LLMMathChain

# llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
# math_tool = Tool(
#         name="Calculator",
#         func=llm_math_chain.run,
#         description="useful for when you need to answer questions about math"
#     ),

# #####==========================================================================> TOOLS
# from langchain.tools.human.tool import HumanInputRun
# from langchain.tools import BaseTool, DuckDuckGoSearchRun










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
from llama_index import GPTSimpleVectorIndex
from llama_index.readers.database import DatabaseReader

from env import settings
from logger import logger



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
            func = lambda *args, **kwargs: self.func(
                *args, **kwargs, get_session=get_session
            )

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
    @tool(
        name="Requests Get",
        description="A portal to the internet. "
        "Use this when you need to get specific content from a website."
        "Input should be a  url (i.e. https://www.google.com)."
        "The output will be the text response of the GET request.",
    )
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


class WineDB(BaseToolSet):
    def __init__(self):
        db = DatabaseReader(
            scheme="postgresql",  # Database Scheme
            host=settings["WINEDB_HOST"],  # Database Host
            port="5432",  # Database Port
            user="alphadom",  # Database User
            password=settings["WINEDB_PASSWORD"],  # Database Password
            dbname="postgres",  # Database Name
        )
        self.columns = ["nameEn", "nameKo", "description"]
        concat_columns = str(",'-',".join([f'"{i}"' for i in self.columns]))
        query = f"""
            SELECT
                Concat({concat_columns})
            FROM wine
        """
        documents = db.load_data(query=query)
        self.index = GPTSimpleVectorIndex(documents)

    @tool(
        name="Wine Recommendation",
        description="A tool to recommend wines based on a user's input. "
        "Inputs are necessary factors for wine recommendations, such as the user's mood today, side dishes to eat with wine, people to drink wine with, what things you want to do, the scent and taste of their favorite wine."
        "The output will be a list of recommended wines."
        "The tool is based on a database of wine reviews, which is stored in a database.",
    )
    def recommend(self, query: str) -> str:
        """Run the tool."""
        results = self.index.query(query)
        wine = "\n".join(
            [
                f"{i}:{j}"
                for i, j in zip(
                    self.columns, results.source_nodes[0].source_text.split("-")
                )
            ]
        )
        output = results.response + "\n\n" + wine

        logger.debug(
            f"\nProcessed WineDB, Input Query: {query} " f"Output Wine: {wine}"
        )

        return output


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

        logger.debug(f"\nProcessed ExitConversation.")

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
    


import signal
from typing import Optional, Tuple

from ptrace.debugger import (
    NewProcessEvent,
    ProcessExecution,
    ProcessExit,
    ProcessSignal,
    PtraceDebugger,
    PtraceProcess,
)
from ptrace.func_call import FunctionCallOptions
from ptrace.syscall import PtraceSyscall
from ptrace.tools import signal_to_exitcode


class SyscallTimeoutException(Exception):
    def __init__(self, pid: int, *args) -> None:
        super().__init__(f"deadline exceeded while waiting syscall for {pid}", *args)


class SyscallTracer:
    def __init__(self, pid: int):
        self.debugger: PtraceDebugger = PtraceDebugger()
        self.pid: int = pid
        self.process: PtraceProcess = None

    def is_waiting(self, syscall: PtraceSyscall) -> bool:
        if syscall.name.startswith("wait"):
            return True
        return False

    def attach(self):
        self.process = self.debugger.addProcess(self.pid, False)

    def detach(self):
        self.process.detach()
        self.debugger.quit()

    def set_timer(self, timeout: int):
        def handler(signum, frame):
            raise SyscallTimeoutException(self.process.pid)

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)

    def reset_timer(self):
        signal.alarm(0)

    def wait_syscall_with_timeout(self, timeout: int):
        self.set_timer(timeout)
        self.process.waitSyscall()
        self.reset_timer()

    def wait_until_stop_or_exit(self) -> Tuple[Optional[int], str]:
        self.process.syscall()
        exitcode = None
        reason = ""
        while True:
            if not self.debugger:
                break

            try:
                self.wait_syscall_with_timeout(30)
            except ProcessExit as event:
                if event.exitcode is not None:
                    exitcode = event.exitcode
                continue
            except ProcessSignal as event:
                event.process.syscall(event.signum)
                exitcode = signal_to_exitcode(event.signum)
                reason = event.reason
                continue
            except NewProcessEvent as event:
                continue
            except ProcessExecution as event:
                continue
            except Exception as e:
                reason = str(e)
                break

            syscall = self.process.syscall_state.event(
                FunctionCallOptions(
                    write_types=False,
                    write_argname=False,
                    string_max_length=300,
                    replace_socketcall=True,
                    write_address=False,
                    max_array_count=20,
                )
            )

            self.process.syscall()

            if syscall is None:
                continue

            if syscall.result:
                continue

        self.reset_timer()

        return exitcode, reason
    

############### => st dout.py

import os
import time
import subprocess
from datetime import datetime
from typing import Callable, Literal, Optional, Union, Tuple

PipeType = Union[Literal["stdout"], Literal["stderr"]]


class StdoutTracer:
    def __init__(
        self,
        process: subprocess.Popen,
        timeout: int = 30,
        interval: int = 0.1,
        on_output: Callable[[PipeType, str], None] = lambda: None,
    ):
        self.process: subprocess.Popen = process
        self.timeout: int = timeout
        self.interval: int = interval
        self.last_output: datetime = None
        self.on_output: Callable[[PipeType, str], None] = on_output

    def nonblock(self):
        os.set_blocking(self.process.stdout.fileno(), False)
        os.set_blocking(self.process.stderr.fileno(), False)

    def get_output(self, pipe: PipeType) -> str:
        output = None
        if pipe == "stdout":
            output = self.process.stdout.read()
        elif pipe == "stderr":
            output = self.process.stderr.read()

        if output:
            decoded = output.decode()
            self.on_output(pipe, decoded)
            self.last_output = datetime.now()
            return decoded
        return ""

    def last_output_passed(self, seconds: int) -> bool:
        return (datetime.now() - self.last_output).seconds > seconds

    def wait_until_stop_or_exit(self) -> Tuple[Optional[int], str]:
        self.nonblock()
        self.last_output = datetime.now()
        output = ""
        exitcode = None
        while True:
            new_stdout = self.get_output("stdout")
            if new_stdout:
                output += new_stdout

            new_stderr = self.get_output("stderr")
            if new_stderr:
                output += new_stderr

            if self.process.poll() is not None:
                exitcode = self.process.poll()
                break

            if self.last_output_passed(self.timeout):
                self.process.kill()
                break

            time.sleep(self.interval)

        return (exitcode, output)

################## => stdout end

import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List

from ansi import ANSI, Color, Style
from env import settings
from logger import logger


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


if __name__ == "__main__":
    import time

    o = Terminal().execute(
        "sleep 1; echo 1; sleep 2; echo 2; sleep 3; echo 3; sleep 10;",
        lambda: ("", None),
    )
    print(o)

    time.sleep(10)  # see if timer has reset
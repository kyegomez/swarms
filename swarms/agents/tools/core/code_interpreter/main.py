import asyncio
import base64
import json
import re
import uuid
from dataclasses import dataclass
from dotenv import load_dotenv
from io import BytesIO
from typing import Any, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

from pydantic import BaseModel, BaseSettings, root_validator

from langchain.agents import AgentExecutor, BaseSingleActionAgent
from langchain.base_language import BaseLanguageModel
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import Callbacks
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models.openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    BaseMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import (
    AgentAction,
    AgentFinish,
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    OutputParserException,
    SystemMessage,
)
from langchain.tools import BaseTool, StructuredTool
from langchain.tools.convert_to_openai import format_tool_to_openai_function



remove_dl_link_prompt = ChatPromptTemplate(
    input_variables=["input_response"],
    messages=[
        SystemMessage(
            content="The user will send you a response and you need to remove the download link from it.\n"
            "Reformat the remaining message so no whitespace or half sentences are still there.\n"
            "If the response does not contain a download link, return the response as is.\n"
        ),
        HumanMessage(
            content="The dataset has been successfully converted to CSV format. You can download the converted file [here](sandbox:/Iris.csv)."
        ),
        AIMessage(content="The dataset has been successfully converted to CSV format."),
        HumanMessagePromptTemplate.from_template("{input_response}"),
    ],
)



async def remove_download_link(
    input_response: str,
    llm: BaseLanguageModel,
) -> str:
    messages = remove_dl_link_prompt.format_prompt(input_response=input_response).to_messages()
    message = await llm.apredict_messages(messages)

    if not isinstance(message, AIMessage):
        raise OutputParserException("Expected an AIMessage")

    return message.content


determine_modifications_prompt = ChatPromptTemplate(
    input_variables=["code"],
    messages=[
        SystemMessage(
            content="The user will input some code and you will need to determine if the code makes any changes to the file system. \n"
            "With changes it means creating new files or modifying exsisting ones.\n"
            "Answer with a function call `determine_modifications` and list them inside.\n"
            "If the code does not make any changes to the file system, still answer with the function call but return an empty list.\n",
        ),
        HumanMessagePromptTemplate.from_template("{code}"),
    ],
)


determine_modifications_function = {
    "name": "determine_modifications",
    "description": "Based on code of the user determine if the code makes any changes to the file system. \n"
    "With changes it means creating new files or modifying exsisting ones.\n",
    "parameters": {
        "type": "object",
        "properties": {
            "modifications": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The filenames that are modified by the code.",
            },
        },
        "required": ["modifications"],
    },
}


determine_modifications_prompt = ChatPromptTemplate(
    input_variables=["code"],
    messages=[
        SystemMessage(
            content="The user will input some code and you will need to determine if the code makes any changes to the file system. \n"
            "With changes it means creating new files or modifying exsisting ones.\n"
            "Answer with a function call `determine_modifications` and list them inside.\n"
            "If the code does not make any changes to the file system, still answer with the function call but return an empty list.\n",
        ),
        HumanMessagePromptTemplate.from_template("{code}"),
    ],
)


determine_modifications_function = {
    "name": "determine_modifications",
    "description": "Based on code of the user determine if the code makes any changes to the file system. \n"
    "With changes it means creating new files or modifying exsisting ones.\n",
    "parameters": {
        "type": "object",
        "properties": {
            "modifications": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The filenames that are modified by the code.",
            },
        },
        "required": ["modifications"],
    },
}


class CodeCallbackHandler(AsyncIteratorCallbackHandler):
    def __init__(self, session: "CodeInterpreterSession"):
        self.session = session
        super().__init__()

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent action."""
        if action.tool == "python":
            await self.session.show_code(
                f"⚙️ Running code: ```python\n{action.tool_input['code']}\n```"  # type: ignore
            )
        else:
            raise ValueError(f"Unknown action: {action.tool}")


async def get_file_modifications(
    code: str,
    llm: BaseLanguageModel,
    retry: int = 2,
) -> Optional[List[str]]:
    if retry < 1:
        return None
    messages = determine_modifications_prompt.format_prompt(code=code).to_messages()
    message = await llm.apredict_messages(messages, functions=[determine_modifications_function])

    if not isinstance(message, AIMessage):
        raise OutputParserException("Expected an AIMessage")

    function_call = message.additional_kwargs.get("function_call", None)

    if function_call is None:
        return await get_file_modifications(code, llm, retry=retry - 1)
    else:
        function_call = json.loads(function_call["arguments"])
        return function_call["modifications"]


system_message = SystemMessage(
    content="""
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. 
As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Assistant is constantly learning and improving, and its capabilities are constantly evolving. 
It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, 
allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

This version of Assistant is called "Code Interpreter" and capable of using a python code interpreter (sandboxed jupyter kernel) to run code. 
The human also maybe thinks this code interpreter is for writing code but it is more for data science, data analysis, and data visualization, file manipulation, and other things that can be done using a jupyter kernel/ipython runtime.
Tell the human if they use the code interpreter incorrectly.
Already installed packages are: (numpy pandas matplotlib seaborn scikit-learn yfinance scipy statsmodels sympy bokeh plotly dash networkx).
If you encounter an error, try again and fix the code.
"""
)


class CodeInterpreterAPISettings(BaseSettings):
    """
    CodeInterpreter API Config
    """

    VERBOSE: bool = False

    CODEBOX_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None


settings = CodeInterpreterAPISettings()

class CodeInput(BaseModel):
    code: str


class FileInput(BaseModel):
    filename: str


class File(BaseModel):
    name: str
    content: bytes

    @classmethod
    def from_path(cls, path: str):
        with open(path, "rb") as f:
            path = path.split("/")[-1]
            return cls(name=path, content=f.read())

    @classmethod
    async def afrom_path(cls, path: str):
        return await asyncio.to_thread(cls.from_path, path)

    @classmethod
    def from_url(cls, url: str):
        import requests  # type: ignore

        r = requests.get(url)
        return cls(name=url.split("/")[-1], content=r.content)

    @classmethod
    async def afrom_url(cls, url: str):
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as r:
                return cls(name=url.split("/")[-1], content=await r.read())

    def save(self, path: str):
        with open(path, "wb") as f:
            f.write(self.content)

    async def asave(self, path: str):
        await asyncio.to_thread(self.save, path)

    def show_image(self):
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            print(
                "Please install it with `pip install codeinterpreterapi[image_support]` to display images."
            )
            exit(1)

        from io import BytesIO

        img_io = BytesIO(self.content)
        img = Image.open(img_io)

        # Convert image to RGB if it's not
        if img.mode not in ('RGB', 'L'):  # L is for greyscale images
            img = img.convert('RGB')

        # Display the image
        try:
            # Try to get the IPython shell if available.
            shell = get_ipython().__class__.__name__  # type: ignore

            # If the shell is ZMQInteractiveShell, it means we're in a Jupyter notebook or similar.
            if shell == 'ZMQInteractiveShell':
                from IPython.display import display
                display(img)
            else:
                # We're not in a Jupyter notebook.
                img.show()
        except NameError:
            # We're probably not in an IPython environment, use PIL's show.
            img.show()



    def __str__(self):
        return self.name

    def __repr__(self):
        return f"File(name={self.name})"


from langchain.schema import HumanMessage, AIMessage  # type: ignore


class UserRequest(HumanMessage):
    files: list[File] = []

    def __str__(self):
        return self.content

    def __repr__(self):
        return f"UserRequest(content={self.content}, files={self.files})"


class CodeInterpreterResponse(AIMessage):
    files: list[File] = []
    # final_code: str = ""  TODO: implement
    # final_output: str = ""  TODO: implement

    def __str__(self):
        return self.content

    def __repr__(self):
        return f"CodeInterpreterResponse(content={self.content}, files={self.files})"



"""
Module implements an agent that uses OpenAI's APIs function enabled API.

This file is a modified version of the original file 
from langchain/agents/openai_functions_agent/base.py.
Credits go to the original authors :)
"""


@dataclass
class _FunctionsAgentAction(AgentAction):
    message_log: List[BaseMessage]


def _convert_agent_action_to_messages(
    agent_action: AgentAction, observation: str
) -> List[BaseMessage]:
    """Convert an agent action to a message.

    This code is used to reconstruct the original AI message from the agent action.

    Args:
        agent_action: Agent action to convert.

    Returns:
        AIMessage that corresponds to the original tool invocation.
    """
    if isinstance(agent_action, _FunctionsAgentAction):
        return agent_action.message_log + [
            _create_function_message(agent_action, observation)
        ]
    else:
        return [AIMessage(content=agent_action.log)]


def _create_function_message(
    agent_action: AgentAction, observation: str
) -> FunctionMessage:
    """Convert agent action and observation into a function message.
    Args:
        agent_action: the tool invocation request from the agent
        observation: the result of the tool invocation
    Returns:
        FunctionMessage that corresponds to the original tool invocation
    """
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except Exception:
            content = str(observation)
    else:
        content = observation
    return FunctionMessage(
        name=agent_action.tool,
        content=content,
    )


def _format_intermediate_steps(
    intermediate_steps: List[Tuple[AgentAction, str]],
) -> List[BaseMessage]:
    """Format intermediate steps.
    Args:
        intermediate_steps: Steps the LLM has taken to date, along with observations
    Returns:
        list of messages to send to the LLM for the next prediction
    """
    messages = []

    for intermediate_step in intermediate_steps:
        agent_action, observation = intermediate_step
        messages.extend(_convert_agent_action_to_messages(agent_action, observation))

    return messages


async def _parse_ai_message(
    message: BaseMessage, llm: BaseLanguageModel
) -> Union[AgentAction, AgentFinish]:
    """Parse an AI message."""
    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected an AI message got {type(message)}")

    function_call = message.additional_kwargs.get("function_call", {})

    if function_call:
        function_call = message.additional_kwargs["function_call"]
        function_name = function_call["name"]
        try:
            _tool_input = json.loads(function_call["arguments"])
        except JSONDecodeError:
            if function_name == "python":
                code = function_call["arguments"]
                _tool_input = {
                    "code": code,
                }
            else:
                raise OutputParserException(
                    f"Could not parse tool input: {function_call} because "
                    f"the `arguments` is not valid JSON."
                )

        # HACK HACK HACK:
        # The code that encodes tool input into Open AI uses a special variable
        # name called `__arg1` to handle old style tools that do not expose a
        # schema and expect a single string argument as an input.
        # We unpack the argument here if it exists.
        # Open AI does not support passing in a JSON array as an argument.
        if "__arg1" in _tool_input:
            tool_input = _tool_input["__arg1"]
        else:
            tool_input = _tool_input

        content_msg = "responded: {content}\n" if message.content else "\n"

        return _FunctionsAgentAction(
            tool=function_name,
            tool_input=tool_input,
            log=f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n",
            message_log=[message],
        )

    return AgentFinish(return_values={"output": message.content}, log=message.content)


class OpenAIFunctionsAgent(BaseSingleActionAgent):
    """An Agent driven by OpenAIs function powered API.

    Args:
        llm: This should be an instance of ChatOpenAI, specifically a model
            that supports using `functions`.
        tools: The tools this agent has access to.
        prompt: The prompt for this agent, should support agent_scratchpad as one
            of the variables. For an easy way to construct this prompt, use
            `OpenAIFunctionsAgent.create_prompt(...)`
    """

    llm: BaseLanguageModel
    tools: Sequence[BaseTool]
    prompt: BasePromptTemplate

    def get_allowed_tools(self) -> List[str]:
        """Get allowed tools."""
        return list([t.name for t in self.tools])

    @root_validator
    def validate_llm(cls, values: dict) -> dict:
        if not isinstance(values["llm"], ChatOpenAI):
            raise ValueError("Only supported with ChatOpenAI models.")
        return values

    @root_validator
    def validate_prompt(cls, values: dict) -> dict:
        prompt: BasePromptTemplate = values["prompt"]
        if "agent_scratchpad" not in prompt.input_variables:
            raise ValueError(
                "`agent_scratchpad` should be one of the variables in the prompt, "
                f"got {prompt.input_variables}"
            )
        return values

    @property
    def input_keys(self) -> List[str]:
        """Get input keys. Input refers to user input here."""
        return ["input"]

    @property
    def functions(self) -> List[dict]:
        return [dict(format_tool_to_openai_function(t)) for t in self.tools]

    def plan(self):
        raise NotImplementedError

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        agent_scratchpad = _format_intermediate_steps(intermediate_steps)
        selected_inputs = {
            k: kwargs[k] for k in self.prompt.input_variables if k != "agent_scratchpad"
        }
        full_inputs = dict(**selected_inputs, agent_scratchpad=agent_scratchpad)
        prompt = self.prompt.format_prompt(**full_inputs)
        messages = prompt.to_messages()
        predicted_message = await self.llm.apredict_messages(
            messages, functions=self.functions, callbacks=callbacks
        )
        agent_decision = await _parse_ai_message(predicted_message, self.llm)
        return agent_decision

    @classmethod
    def create_prompt(
        cls,
        system_message: Optional[SystemMessage] = SystemMessage(
            content="You are a helpful AI assistant."
        ),
        extra_prompt_messages: Optional[List[BaseMessagePromptTemplate]] = None,
    ) -> BasePromptTemplate:
        """Create prompt for this agent.

        Args:
            system_message: Message to use as the system message that will be the
                first in the prompt.
            extra_prompt_messages: Prompt messages that will be placed between the
                system message and the new human input.

        Returns:
            A prompt template to pass into this agent.
        """
        _prompts = extra_prompt_messages or []
        messages: List[Union[BaseMessagePromptTemplate, BaseMessage]]
        if system_message:
            messages = [system_message]
        else:
            messages = []

        messages.extend(
            [
                *_prompts,
                HumanMessagePromptTemplate.from_template("{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        return ChatPromptTemplate(messages=messages)  # type: ignore

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        extra_prompt_messages: Optional[List[BaseMessagePromptTemplate]] = None,
        system_message: Optional[SystemMessage] = SystemMessage(
            content="You are a helpful AI assistant."
        ),
        **kwargs: Any,
    ) -> BaseSingleActionAgent:
        """Construct an agent from an LLM and tools."""
        if not isinstance(llm, ChatOpenAI):
            raise ValueError("Only supported with ChatOpenAI models.")
        prompt = cls.create_prompt(
            extra_prompt_messages=extra_prompt_messages,
            system_message=system_message,
        )
        return cls(
            llm=llm,
            prompt=prompt,
            tools=tools,
            callback_manager=callback_manager,  # type: ignore
            **kwargs,
        )
    



class CodeInterpreterSession:
    def __init__(
        self,
        model=None,
        openai_api_key=settings.OPENAI_API_KEY,
        verbose=settings.VERBOSE,
        tools: list[BaseTool] = None
    ) -> None:
        self.codebox = CodeBox()
        self.verbose = verbose
        self.tools: list[BaseTool] = self._tools(tools)
        self.llm: BaseChatModel = self._llm(model, openai_api_key)
        self.agent_executor: AgentExecutor = self._agent_executor()
        self.input_files: list[File] = []
        self.output_files: list[File] = []

    async def astart(self) -> None:
        await self.codebox.astart()

    def _tools(self, additional_tools: list[BaseTool] = None) -> list[BaseTool]:
        additional_tools = additional_tools or []
        return additional_tools + [
            StructuredTool(
                name="python",
                description=
                # TODO: variables as context to the agent
                # TODO: current files as context to the agent
                "Input a string of code to a python interpreter (jupyter kernel). "
                "Variables are preserved between runs. ",
                func=self.run_handler,
                coroutine=self.arun_handler,
                args_schema=CodeInput,
            ),
        ]

    def _llm(self, model: Optional[str] = None, openai_api_key: Optional[str] = None) -> BaseChatModel:
        if model is None:
            model = "gpt-4"

        if openai_api_key is None:
            raise ValueError(
                "OpenAI API key missing. Set OPENAI_API_KEY env variable or pass `openai_api_key` to session."
            )

        return ChatOpenAI(
            temperature=0.03,
            model=model,
            openai_api_key=openai_api_key,
            max_retries=3,
            request_timeout=60 * 3,
        )  # type: ignore

    def _agent(self) -> BaseSingleActionAgent:
        return OpenAIFunctionsAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools,
            system_message=code_interpreter_system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name="memory")],
        )

    def _agent_executor(self) -> AgentExecutor:
        return AgentExecutor.from_agent_and_tools(
            agent=self._agent(),
            callbacks=[CodeCallbackHandler(self)],
            max_iterations=9,
            tools=self.tools,
            verbose=self.verbose,
            memory=ConversationBufferMemory(memory_key="memory", return_messages=True),
        )

    async def show_code(self, code: str) -> None:
        """Callback function to show code to the user."""
        if self.verbose:
            print(code)

    def run_handler(self, code: str):
        raise NotImplementedError("Use arun_handler for now.")

    async def arun_handler(self, code: str):
        """Run code in container and send the output to the user"""
        output: CodeBoxOutput = await self.codebox.arun(code)

        if not isinstance(output.content, str):
            raise TypeError("Expected output.content to be a string.")

        if output.type == "image/png":
            filename = f"image-{uuid.uuid4()}.png"
            file_buffer = BytesIO(base64.b64decode(output.content))
            file_buffer.name = filename
            self.output_files.append(File(name=filename, content=file_buffer.read()))
            return f"Image {filename} got send to the user."

        elif output.type == "error":
            if "ModuleNotFoundError" in output.content:
                if package := re.search(
                    r"ModuleNotFoundError: No module named '(.*)'", output.content
                ):
                    await self.codebox.ainstall(package.group(1))
                    return f"{package.group(1)} was missing but got installed now. Please try again."
            else: pass
                # TODO: preanalyze error to optimize next code generation
            if self.verbose:
                print("Error:", output.content)

        elif modifications := await get_file_modifications(code, self.llm):
            for filename in modifications:
                if filename in [file.name for file in self.input_files]:
                    continue
                fileb = await self.codebox.adownload(filename)
                if not fileb.content:
                    continue
                file_buffer = BytesIO(fileb.content)
                file_buffer.name = filename
                self.output_files.append(
                    File(name=filename, content=file_buffer.read())
                )

        return output.content

    async def input_handler(self, request: UserRequest):
        if not request.files:
            return
        if not request.content:
            request.content = (
                "I uploaded, just text me back and confirm that you got the file(s)."
            )
        request.content += "\n**The user uploaded the following files: **\n"
        for file in request.files:
            self.input_files.append(file)
            request.content += f"[Attachment: {file.name}]\n"
            await self.codebox.aupload(file.name, file.content)
        request.content += "**File(s) are now available in the cwd. **\n"

    async def output_handler(self, final_response: str) -> CodeInterpreterResponse:
        """Embed images in the response"""
        for file in self.output_files:
            if str(file.name) in final_response:
                # rm ![Any](file.name) from the response
                final_response = re.sub(rf"\n\n!\[.*\]\(.*\)", "", final_response)

        if self.output_files and re.search(rf"\n\[.*\]\(.*\)", final_response):
            final_response = await remove_download_link(final_response, self.llm)

        return CodeInterpreterResponse(content=final_response, files=self.output_files)

    async def generate_response(
        self,
        user_msg: str,
        files: list[File] = [],
        detailed_error: bool = False,
    ) -> CodeInterpreterResponse:
        """Generate a Code Interpreter response based on the user's input."""
        user_request = UserRequest(content=user_msg, files=files)
        try:
            await self.input_handler(user_request)
            response = await self.agent_executor.arun(input=user_request.content)
            return await self.output_handler(response)
        except Exception as e:
            if self.verbose:
                import traceback

                traceback.print_exc()
            if detailed_error:
                return CodeInterpreterResponse(
                    content=f"Error in CodeInterpreterSession: {e.__class__.__name__}  - {e}"
                )
            else:
                return CodeInterpreterResponse(
                    content="Sorry, something went while generating your response."
                    "Please try again or restart the session."
                )

    async def is_running(self) -> bool:
        return await self.codebox.astatus() == "running"

    async def astop(self) -> None:
        await self.codebox.astop()

    async def __aenter__(self) -> "CodeInterpreterSession":
        await self.astart()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.astop()
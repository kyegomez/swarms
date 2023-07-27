from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from celery import Task

# from ansi import ANSI, Color, Style, dim_multiline
from swarms.utils.main import ANSI, Color, Style, dim_multiline
from swarms.utils.logger import logger


class EVALCallbackHandler(BaseCallbackHandler):
    @property
    def ignore_llm(self) -> bool:
        return False

    def set_parser(self, parser) -> None:
        self.parser = parser

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        text = response.generations[0][0].text

        parsed = self.parser.parse_all(text)

        logger.info(ANSI("Plan").to(Color.blue().bright()) + ": " + parsed["plan"])
        logger.info(ANSI("What I Did").to(Color.blue()) + ": " + parsed["what_i_did"])
        logger.info(
            ANSI("Action").to(Color.cyan())
            + ": "
            + ANSI(parsed["action"]).to(Style.bold())
        )
        logger.info(
            ANSI("Input").to(Color.cyan())
            + ": "
            + dim_multiline(parsed["action_input"])
        )

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        logger.info(ANSI(f"on_llm_new_token {token}").to(Color.green(), Style.italic()))

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        logger.info(ANSI("Entering new chain.").to(Color.green(), Style.italic()))
        logger.info(ANSI("Prompted Text").to(Color.yellow()) + f': {inputs["input"]}\n')

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        logger.info(ANSI("Finished chain.").to(Color.green(), Style.italic()))

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        logger.error(
            ANSI("Chain Error").to(Color.red()) + ": " + dim_multiline(str(error))
        )

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        pass

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        logger.info(
            ANSI("Observation").to(Color.magenta()) + ": " + dim_multiline(output)
        )
        logger.info(ANSI("Thinking...").to(Color.green(), Style.italic()))

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        logger.error(ANSI("Tool Error").to(Color.red()) + f": {error}")

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Optional[str],
    ) -> None:
        pass

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        logger.info(
            ANSI("Final Answer").to(Color.yellow())
            + ": "
            + dim_multiline(finish.return_values.get("output", ""))
        )


class ExecutionTracingCallbackHandler(BaseCallbackHandler):
    def __init__(self, execution: Task):
        self.execution = execution
        self.index = 0

    def set_parser(self, parser) -> None:
        self.parser = parser

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        text = response.generations[0][0].text
        parsed = self.parser.parse_all(text)
        self.index += 1
        parsed["index"] = self.index
        self.execution.update_state(state="LLM_END", meta=parsed)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self.execution.update_state(state="CHAIN_ERROR", meta={"error": str(error)})

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        pass

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        previous = self.execution.AsyncResult(self.execution.request.id)
        self.execution.update_state(
            state="TOOL_END", meta={**previous.info, "observation": output}
        )

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        previous = self.execution.AsyncResult(self.execution.request.id)
        self.execution.update_state(
            state="TOOL_ERROR", meta={**previous.info, "error": str(error)}
        )

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Optional[str],
    ) -> None:
        pass

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        pass
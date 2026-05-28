import argparse
import os
import sys
from typing import Any, List

from dotenv import load_dotenv

from swarms import Agent
from swarms.utils.litellm_wrapper import LiteLLM


def _pick_model(explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model
    return os.getenv(
        "LIVE_TEST_MODEL",
        os.getenv("SMOKE_MODEL", "gemini/gemini-2.0-flash"),
    )


def _find_present_keys() -> List[str]:
    key_names = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
    ]
    return [name for name in key_names if os.getenv(name)]


def run_live_test(
    model_name: str,
    task: str,
    max_tokens: int,
    timeout_seconds: int,
    require_streaming: bool,
) -> int:
    streamed_tokens: List[str] = []

    def streaming_callback(token: Any) -> None:
        # Detailed stream mode emits token metadata dictionaries.
        if isinstance(token, dict):
            token_text = token.get("token")
            if token_text:
                streamed_tokens.append(str(token_text))
            return
        if token:
            streamed_tokens.append(str(token))

    # Sonnet 4 requests in this path require temperature=1.
    temperature = (
        1.0 if "claude-sonnet-4" in model_name.lower() else 0.1
    )

    llm = LiteLLM(
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout_seconds,
    )

    agent = Agent(
        llm=llm,
        model_name=model_name,
        temperature=temperature,
        stream=False,
        max_loops="auto",
        timeout=timeout_seconds,
        print_on=False,
        verbose=False,
        reasoning_prompt_on=False,
        selected_tools=[
            "create_plan",
            "think",
            "subtask_done",
            "complete_task",
        ],
    )

    # Keep autonomous planning/tool calls non-streaming, then stream only
    # during final summary generation where this bug fix is targeted.
    original_generate_final_summary = agent._generate_final_summary

    def generate_final_summary_with_streaming(*, streaming_callback=None):
        previous_agent_stream = agent.stream
        previous_llm_stream = getattr(agent.llm, "stream", False)
        previous_tools = agent.tools
        previous_llm_tools = getattr(
            agent.llm, "tools_list_dictionary", None
        )
        previous_llm_tool_choice = getattr(
            agent.llm, "tool_choice", None
        )
        agent.stream = True
        agent.tools = None
        if hasattr(agent.llm, "stream"):
            agent.llm.stream = True
        if hasattr(agent.llm, "tools_list_dictionary"):
            agent.llm.tools_list_dictionary = None
        if hasattr(agent.llm, "tool_choice"):
            agent.llm.tool_choice = None
        try:
            return original_generate_final_summary(
                streaming_callback=streaming_callback
            )
        finally:
            agent.stream = previous_agent_stream
            agent.tools = previous_tools
            if hasattr(agent.llm, "stream"):
                agent.llm.stream = previous_llm_stream
            if hasattr(agent.llm, "tools_list_dictionary"):
                agent.llm.tools_list_dictionary = (
                    previous_llm_tools
                )
            if hasattr(agent.llm, "tool_choice"):
                agent.llm.tool_choice = (
                    previous_llm_tool_choice
                )

    agent._generate_final_summary = (
        generate_final_summary_with_streaming
    )

    print(f"RUN_MODE: autonomous")
    print(f"TIMEOUT_SECONDS: {timeout_seconds}")

    run_error: Exception | None = None
    result: Any = None
    try:
        result = agent.run(
            task=task, streaming_callback=streaming_callback
        )
    except Exception as exc:
        run_error = exc

    print(f"MODEL: {model_name}")
    print(f"STREAM_TOKEN_COUNT: {len(streamed_tokens)}")
    print("RESULT_PREVIEW:")
    print(str(result)[:600] if result is not None else "none")

    if require_streaming and len(streamed_tokens) == 0:
        print(
            "LIVE_TEST_ERROR: Streaming callback produced zero tokens; "
            "treating as failure."
        )
        return 3

    if run_error is not None:
        print("LIVE_TEST_ERROR: Agent run failed.")
        print(str(run_error))
        return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Permanent live test for autonomous summary streaming callback."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Provider/model id, for example gemini/gemini-2.0-flash",
    )
    parser.add_argument(
        "--check-env-only",
        action="store_true",
        help="Only validate env keys and selected model, do not call provider.",
    )
    parser.add_argument(
        "--task",
        default=(
            "Write exactly one short sentence: Streaming callback verified."
        ),
        help="Task prompt to run in autonomous mode.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="LLM max tokens for each model call.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Network/request timeout seconds for both model and agent call path.",
    )
    parser.add_argument(
        "--allow-zero-stream-tokens",
        action="store_true",
        help="Do not fail when stream token count is zero.",
    )
    args = parser.parse_args()

    load_dotenv()

    model_name = _pick_model(args.model)
    present_keys = _find_present_keys()

    print(f"SELECTED_MODEL: {model_name}")
    print(f"PRESENT_KEYS: {present_keys if present_keys else 'none'}")

    if args.check_env_only:
        return 0

    if not present_keys:
        print("No provider API key found in environment. Aborting live run.")
        return 2

    try:
        return run_live_test(
            model_name=model_name,
            task=args.task,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
            require_streaming=(
                not args.allow_zero_stream_tokens
            ),
        )
    except Exception as exc:
        print("LIVE_TEST_ERROR:")
        print(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())

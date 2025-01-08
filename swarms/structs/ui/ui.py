import os
from dotenv import load_dotenv
from typing import AsyncGenerator, List, Dict, Any, Tuple, Optional
import json
import time
import asyncio
import gradio as gr
from swarms.structs.agent import Agent
from swarms.structs.swarm_router import SwarmRouter
from swarms.utils.loguru_logger import initialize_logger
import re
import csv  # Import the csv module for csv parsing
from swarms.utils.litellm_wrapper import LiteLLM
from litellm import models_by_provider
from dotenv import set_key, find_dotenv
import logging  # Import the logging module
import litellm # Import litellm exception

# Initialize logger
load_dotenv()

# Initialize logger
logger = initialize_logger(log_folder="swarm_ui")

# Define the path to agent_prompts.json
PROMPT_JSON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "agent_prompts.json"
)
logger.info(f"Loading prompts from: {PROMPT_JSON_PATH}")

# Load prompts first so its available for create_app
def load_prompts_from_json() -> Dict[str, str]:
    try:
        if not os.path.exists(PROMPT_JSON_PATH):
            # Load default prompts
            return {
                "Agent-Data_Extractor": "You are a data extraction agent...",
                "Agent-Summarizer": "You are a summarization agent...",
                "Agent-Onboarding_Agent": "You are an onboarding agent...",
            }

        with open(PROMPT_JSON_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # Load default prompts
                return {
                    "Agent-Data_Extractor": "You are a data extraction agent...",
                    "Agent-Summarizer": "You are a summarization agent...",
                    "Agent-Onboarding_Agent": "You are an onboarding agent...",
                }

            if not isinstance(data, dict):
                # Load default prompts
                return {
                    "Agent-Data_Extractor": "You are a data extraction agent...",
                    "Agent-Summarizer": "You are a summarization agent...",
                    "Agent-Onboarding_Agent": "You are an onboarding agent...",
                }

            prompts = {}
            for agent_name, details in data.items():
                if (
                    not isinstance(details, dict)
                    or "system_prompt" not in details
                ):
                    continue

                prompts[agent_name] = details["system_prompt"]

            if not prompts:
                # Load default prompts
                return {
                    "Agent-Data_Extractor": "You are a data extraction agent...",
                    "Agent-Summarizer": "You are a summarization agent...",
                    "Agent-Onboarding_Agent": "You are an onboarding agent...",
                }

            return prompts

    except Exception:
        # Load default prompts
        return {
            "Agent-Data_Extractor": "You are a data extraction agent...",
            "Agent-Summarizer": "You are a summarization agent...",
            "Agent-Onboarding_Agent": "You are an onboarding agent...",
        }

AGENT_PROMPTS = load_prompts_from_json()

api_keys = {}

def initialize_agents(
    dynamic_temp: float,
    agent_keys: List[str],
    model_name: str,
    provider: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> List[Agent]:
    logger.info("Initializing agents...")
    agents = []
    seen_names = set()
    try:
        for agent_key in agent_keys:
            if agent_key not in AGENT_PROMPTS:
                raise ValueError(f"Invalid agent key: {agent_key}")

            agent_prompt = AGENT_PROMPTS[agent_key]
            agent_name = agent_key

            # Ensure unique agent names
            base_name = agent_name
            counter = 1
            while agent_name in seen_names:
                agent_name = f"{base_name}_{counter}"
                counter += 1
            seen_names.add(agent_name)

            # Set API key using os.environ temporarily
            if provider == "openai":
                os.environ["OPENAI_API_KEY"] = api_key
            elif provider == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = api_key
            elif provider == "cohere":
                os.environ["COHERE_API_KEY"] = api_key
            elif provider == "gemini":
                os.environ["GEMINI_API_KEY"] = api_key
            elif provider == "mistral":
                os.environ["MISTRAL_API_KEY"] = api_key
            elif provider == "groq":
                os.environ["GROQ_API_KEY"] = api_key
            elif provider == "perplexity":
                os.environ["PERPLEXITY_API_KEY"] = api_key
            # Add other providers and their environment variable names as needed

            # Create LiteLLM instance (Now it will read from os.environ)
            llm = LiteLLM(
                model_name=model_name,
                system_prompt=agent_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            agent = Agent(
                agent_name=agent_name,
                system_prompt=agent_prompt,
                llm=llm,
                max_loops=1,
                autosave=True,
                verbose=True,
                dynamic_temperature_enabled=True,
                saved_state_path=f"agent_{agent_name}.json",
                user_name="pe_firm",
                retry_attempts=1,
                context_length=200000,
                output_type="string",  # here is the output type which is string
                temperature=dynamic_temp,
            )
            print(f"Agent created: {agent.agent_name}")
            agents.append(agent)

        logger.info(f"Agents initialized successfully: {[agent.agent_name for agent in agents]}")
        return agents
    except Exception as e:
        logger.error(f"Error initializing agents: {e}", exc_info=True)
        raise

def validate_flow(flow, agents_dict):
    logger.info(f"Validating flow: {flow}")
    agent_names = flow.split("->")
    for agent in agent_names:
        agent = agent.strip()
        if agent not in agents_dict:
            logger.error(f"Agent '{agent}' specified in the flow does not exist.")
            raise ValueError(
                f"Agent '{agent}' specified in the flow does not exist."
            )
    logger.info(f"Flow validated successfully: {flow}")

class TaskExecutionError(Exception):
    """Custom exception for task execution errors."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"TaskExecutionError: {self.message}"
    
async def execute_task(
    task: str,
    max_loops: int,
    dynamic_temp: float,
    swarm_type: str,
    agent_keys: List[str],
    flow: str = None,
    model_name: str = "gpt-4o",
    provider: str = "openai",
    api_key: str = None,
    temperature: float = 0.5,
    max_tokens: int = 4000,
    agents: dict = None,
    log_display=None,
    error_display=None
) -> AsyncGenerator[Tuple[Any, Optional["SwarmRouter"], str], None]: # Changed the return type here
    logger.info(f"Executing task: {task} with swarm type: {swarm_type}")
    try:
        if not task:
            logger.error("Task description is missing.")
            yield  "Please provide a task description.",  gr.update(visible=True), ""
            return
        if not agent_keys:
            logger.error("No agents selected.")
            yield  "Please select at least one agent.",  gr.update(visible=True), ""
            return
        if not provider:
            logger.error("Provider is missing.")
            yield "Please select a provider.", gr.update(visible=True), ""
            return
        if not model_name:
           logger.error("Model is missing.")
           yield  "Please select a model.",  gr.update(visible=True), ""
           return
        if not api_key:
            logger.error("API Key is missing.")
            yield  "Please enter an API Key.",  gr.update(visible=True), ""
            return

        # Initialize agents
        try:
            if not agents:
                agents = initialize_agents(
                    dynamic_temp,
                    agent_keys,
                    model_name,
                    provider,
                    api_key,
                    temperature,
                    max_tokens,
                )
        except Exception as e:
             logger.error(f"Error initializing agents: {e}", exc_info=True)
             yield f"Error initializing agents: {e}", gr.update(visible=True), ""
             return

        # Swarm-specific configurations
        router_kwargs = {
            "name": "multi-agent-workflow",
            "description": f"Executing {swarm_type} workflow",
            "max_loops": max_loops,
            "agents": list(agents.values()),
            "autosave": True,
            "return_json": True,
            "output_type": "string", # Default output type
            "swarm_type": swarm_type,  # Pass swarm_type here
        }

        if swarm_type == "AgentRearrange":
            if not flow:
                logger.error("Flow configuration is missing for AgentRearrange.")
                yield "Flow configuration is required for AgentRearrange", gr.update(visible=True), ""
                return

            # Generate unique agent names in the flow
            flow_agents = []
            used_agent_names = set()
            for agent_key in flow.split("->"):
                agent_key = agent_key.strip()
                base_agent_name = agent_key
                count = 1
                while agent_key in used_agent_names:
                    agent_key = f"{base_agent_name}_{count}"
                    count += 1
                used_agent_names.add(agent_key)
                flow_agents.append(agent_key)

            # Update the flow string with unique names
            flow = " -> ".join(flow_agents)
            logger.info(f"Updated Flow string: {flow}")
            router_kwargs["flow"] = flow
            router_kwargs["output_type"] = "string"  # Changed output type here

        if swarm_type == "MixtureOfAgents":
            if len(agents) < 2:
                logger.error("MixtureOfAgents requires at least 2 agents.")
                yield "MixtureOfAgents requires at least 2 agents", gr.update(visible=True), ""
                return

        if swarm_type == "SequentialWorkflow":
            if len(agents) < 2:
                logger.error("SequentialWorkflow requires at least 2 agents.")
                yield "SequentialWorkflow requires at least 2 agents",  gr.update(visible=True), ""
                return

        if swarm_type == "ConcurrentWorkflow":
            pass

        if swarm_type == "SpreadSheetSwarm":
             pass

        if swarm_type == "auto":
             pass

        # Create and execute SwarmRouter
        try:
            timeout = (
                450 if swarm_type != "SpreadSheetSwarm" else 900
            )  # SpreadSheetSwarm will have different timeout.

            if swarm_type == "AgentRearrange":
                 from swarms.structs.rearrange import AgentRearrange
                 router = AgentRearrange(
                    agents=list(agents.values()),
                    flow=flow,
                    max_loops=max_loops,
                    name="multi-agent-workflow",
                    description=f"Executing {swarm_type} workflow",
                    # autosave=True,
                    return_json=True,
                    output_type="string",  # Changed output type according to agent rearrange
                 )
                 result = router(task)  # Changed run method
                 logger.info(f"AgentRearrange task executed successfully.")
                 yield result, None, ""
                 return
            
            # For other swarm types use the SwarmRouter and its run method
            router = SwarmRouter(**router_kwargs)  # Initialize SwarmRouter
            if swarm_type == "ConcurrentWorkflow":
                async def run_agent_task(agent, task_):
                    return agent.run(task_)

                tasks = [
                    run_agent_task(agent, task)
                    for agent in list(agents.values())
                ]
                responses = await asyncio.gather(*tasks)
                result = {}
                for agent, response in zip(list(agents.values()), responses):
                    result[agent.agent_name] = response
                
                # Convert the result to JSON string for parsing
                result = json.dumps(
                    {
                        "input" : {
                        "swarm_id" : "concurrent_workflow_swarm_id",
                        "name" : "ConcurrentWorkflow",
                        "flow" : "->".join([agent.agent_name for agent in list(agents.values())])
                        },
                        "time" : time.time(),
                        "outputs" : [
                         {
                            "agent_name": agent_name,
                            "steps" : [{"role":"assistant", "content":response}]
                         } for agent_name, response in result.items()
                        ]
                    }
                )
                logger.info(f"ConcurrentWorkflow task executed successfully.")
                yield result, None, ""
                return
            elif swarm_type == "auto":
                 result = await asyncio.wait_for(
                    asyncio.to_thread(router.run, task),
                    timeout=timeout
                 )
                 if isinstance(result,dict):
                     result = json.dumps(
                         {
                            "input" : {
                                "swarm_id" : "auto_swarm_id",
                                "name" : "AutoSwarm",
                                "flow" : "->".join([agent.agent_name for agent in list(agents.values())])
                            },
                            "time" : time.time(),
                             "outputs" : [
                                {
                                    "agent_name": agent.agent_name,
                                    "steps" : [{"role":"assistant", "content":response}]
                                } for agent, response in result.items()
                            ]
                         }
                     )
                 elif isinstance(result, str):
                     result = json.dumps(
                         {
                            "input" : {
                                "swarm_id" : "auto_swarm_id",
                                "name" : "AutoSwarm",
                                "flow" : "->".join([agent.agent_name for agent in list(agents.values())])
                            },
                            "time" : time.time(),
                             "outputs" : [
                                {
                                    "agent_name": "auto",
                                    "steps" : [{"role":"assistant", "content":result}]
                                }
                            ]
                         }
                    )
                 else :
                     logger.error("Auto Swarm returned an unexpected type")
                     yield "Error : Auto Swarm returned an unexpected type", gr.update(visible=True), ""
                     return
                 logger.info(f"Auto task executed successfully.")
                 yield result, None, ""
                 return
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(router.run, task),
                    timeout=timeout
                )
                logger.info(f"{swarm_type} task executed successfully.")
                yield result, None, ""
                return
        except asyncio.TimeoutError as e:
             logger.error(f"Task execution timed out after {timeout} seconds", exc_info=True)
             yield f"Task execution timed out after {timeout} seconds", gr.update(visible=True), ""
             return
        except litellm.exceptions.APIError as e: # Catch litellm APIError
            logger.error(f"LiteLLM API Error: {e}", exc_info=True)
            yield f"LiteLLM API Error: {e}", gr.update(visible=True), ""
            return
        except litellm.exceptions.AuthenticationError as e: # Catch litellm AuthenticationError
            logger.error(f"LiteLLM Authentication Error: {e}", exc_info=True)
            yield f"LiteLLM Authentication Error: {e}", gr.update(visible=True), ""
            return
        except Exception as e:
            logger.error(f"Error executing task: {e}", exc_info=True)
            yield f"Error executing task: {e}",  gr.update(visible=True), ""
            return

    except TaskExecutionError as e:
        logger.error(f"Task execution error: {e}")
        yield str(e),  gr.update(visible=True), ""
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        yield f"An unexpected error occurred: {e}",  gr.update(visible=True), ""
        return
    finally:
        logger.info(f"Task execution finished for: {task} with swarm type: {swarm_type}")

def format_output(data:Optional[str], swarm_type:str, error_display=None) -> str:
    if data is None:
       return "Error : No output from the swarm."
    if swarm_type == "AgentRearrange":
        return parse_agent_rearrange_output(data, error_display)
    elif swarm_type == "MixtureOfAgents":
         return parse_mixture_of_agents_output(data, error_display)
    elif swarm_type in ["SequentialWorkflow", "ConcurrentWorkflow"]:
         return parse_sequential_workflow_output(data, error_display)
    elif swarm_type == "SpreadSheetSwarm":
         if os.path.exists(data):
            return parse_spreadsheet_swarm_output(data, error_display)
         else:
              return data # Directly return JSON response
    elif swarm_type == "auto":
        return parse_auto_swarm_output(data, error_display)
    else:
        return "Unsupported swarm type."

def parse_mixture_of_agents_data(data: dict, error_display=None) -> str:
    """Parses the MixtureOfAgents output data and formats it for display."""
    logger.info("Parsing MixtureOfAgents data within Auto Swarm output...")

    try:
        output = ""
        if "InputConfig" in data and isinstance(data["InputConfig"], dict):
            input_config = data["InputConfig"]
            output += f"Mixture of Agents Workflow Details\n\n"
            output += f"Name: `{input_config.get('name', 'N/A')}`\n"
            output += (
                f"Description:"
                f" `{input_config.get('description', 'N/A')}`\n\n---\n"
            )
            output += f"Agent Task Execution\n\n"

            for agent in input_config.get("agents", []):
                output += (
                    f"Agent: `{agent.get('agent_name', 'N/A')}`\n"
                )

        if "normal_agent_outputs" in data and isinstance(
            data["normal_agent_outputs"], list
        ):
            for i, agent_output in enumerate(
                data["normal_agent_outputs"], start=3
            ):
                agent_name = agent_output.get("agent_name", "N/A")
                output += f"Run {(3 - i)} (Agent: `{agent_name}`)\n\n"
                for j, step in enumerate(
                    agent_output.get("steps", []), start=3
                ):
                    if (
                        isinstance(step, dict)
                        and "role" in step
                        and "content" in step
                        and step["role"].strip() != "System:"
                    ):
                        content = step["content"]
                        output += f"Step {(3 - j)}: \n"
                        output += f"Response:\n {content}\n\n"

        if "aggregator_agent_summary" in data:
            output += (
                f"\nAggregated Summary :\n"
                f"{data['aggregator_agent_summary']}\n{'=' * 50}\n"
            )

        logger.info("MixtureOfAgents data parsed successfully within Auto Swarm.")
        return output

    except Exception as e:
        logger.error(
            f"Error during parsing MixtureOfAgents data within Auto Swarm: {e}",
            exc_info=True,
        )
        return f"Error during parsing: {str(e)}"

def parse_auto_swarm_output(data: Optional[str], error_display=None) -> str:
    """Parses the auto swarm output string and formats it for display."""
    logger.info("Parsing Auto Swarm output...")
    if data is None:
        logger.error("No data provided for parsing Auto Swarm output.")
        return "Error: No data provided for parsing."

    print(f"Raw data received for parsing:\n{data}")  # Debug: Print raw data

    try:
        parsed_data = json.loads(data)
        errors = []

        # Basic structure validation
        if (
            "input" not in parsed_data
            or not isinstance(parsed_data.get("input"), dict)
        ):
            errors.append(
                "Error: 'input' data is missing or not a dictionary."
            )
        else:
            if "swarm_id" not in parsed_data["input"]:
                errors.append(
                    "Error: 'swarm_id' key is missing in the 'input'."
                )
            if "name" not in parsed_data["input"]:
                errors.append(
                    "Error: 'name' key is missing in the 'input'."
                )
            if "flow" not in parsed_data["input"]:
                errors.append(
                    "Error: 'flow' key is missing in the 'input'."
                )

        if "time" not in parsed_data:
            errors.append("Error: 'time' key is missing.")

        if errors:
            logger.error(
                f"Errors found while parsing Auto Swarm output: {errors}"
            )
            return "\n".join(errors)

        swarm_id = parsed_data["input"]["swarm_id"]
        swarm_name = parsed_data["input"]["name"]
        agent_flow = parsed_data["input"]["flow"]
        overall_time = parsed_data["time"]

        output = f"Workflow Execution Details\n\n"
        output += f"Swarm ID: `{swarm_id}`\n"
        output += f"Swarm Name: `{swarm_name}`\n"
        output += f"Agent Flow: `{agent_flow}`\n\n---\n"
        output += f"Agent Task Execution\n\n"

        # Handle nested MixtureOfAgents data or other swarm type data
        if (
            "outputs" in parsed_data
            and isinstance(parsed_data["outputs"], list)
            and parsed_data["outputs"]
            and isinstance(parsed_data["outputs"][0], dict)
        ):
            if parsed_data["outputs"][0].get("agent_name") == "auto":
                mixture_data = parsed_data["outputs"][0].get("steps", [])
                if mixture_data and isinstance(mixture_data[0], dict) and "content" in mixture_data[0]:
                    try:
                        mixture_content = json.loads(mixture_data[0]["content"])
                        output += parse_mixture_of_agents_data(mixture_content)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding nested MixtureOfAgents data: {e}", exc_info=True)
                        return f"Error decoding nested MixtureOfAgents data: {e}"
            else:
                for i, agent_output in enumerate(parsed_data["outputs"], start=3):
                    if not isinstance(agent_output, dict):
                        errors.append(f"Error: Agent output at index {i} is not a dictionary")
                        continue
                    if "agent_name" not in agent_output:
                        errors.append(f"Error: 'agent_name' key is missing at index {i}")
                        continue
                    if "steps" not in agent_output:
                        errors.append(f"Error: 'steps' key is missing at index {i}")
                        continue
                    if agent_output["steps"] is None:
                        errors.append(f"Error: 'steps' data is None at index {i}")
                        continue
                    if not isinstance(agent_output["steps"], list):
                        errors.append(f"Error: 'steps' data is not a list at index {i}")
                        continue

                    
                    agent_name = agent_output["agent_name"]
                    output += f"Run {(3-i)} (Agent: `{agent_name}`)\n\n"

                    # Iterate over steps
                    for j, step in enumerate(agent_output["steps"], start=3):
                        if not isinstance(step, dict):
                            errors.append(f"Error: step at index {j} is not a dictionary at {i} agent output.")
                            continue
                        if step is None:
                            errors.append(f"Error: step at index {j} is None at {i} agent output")
                            continue

                        if "role" not in step:
                            errors.append(f"Error: 'role' key missing at step {j} at {i} agent output.")
                            continue
                        
                        if "content" not in step:
                            errors.append(f"Error: 'content' key missing at step {j} at {i} agent output.")
                            continue
                        
                        if step["role"].strip() != "System:":  # Filter out system prompts
                            content = step["content"]
                            output += f"Step {(3-j)}:\n"
                            output += f"Response : {content}\n\n"
        else:
            logger.error("Error: 'outputs' data is not in the expected format.")
            return "Error: 'outputs' data is not in the expected format."

        output += f"Overall Completion Time: `{overall_time}`"

        if errors:
            logger.error(
                f"Errors found while parsing Auto Swarm output: {errors}"
            )
            return "\n".join(errors)

        logger.info("Auto Swarm output parsed successfully.")
        return output

    except json.JSONDecodeError as e:
        logger.error(
            f"Error during parsing Auto Swarm output: {e}", exc_info=True
        )
        return f"Error during parsing json.JSONDecodeError: {e}"

    except Exception as e:
        logger.error(
            f"Error during parsing Auto Swarm output: {e}", exc_info=True
        )
        return f"Error during parsing: {str(e)}"



def parse_agent_rearrange_output(data: Optional[str], error_display=None) -> str:
    """
    Parses the AgentRearrange output string and formats it for display.
    """
    logger.info("Parsing AgentRearrange output...")
    if data is None:
        logger.error("No data provided for parsing AgentRearrange output.")
        return "Error: No data provided for parsing."

    print(
        f"Raw data received for parsing:\n{data}"
    )  # Debug: Print raw data

    try:
        parsed_data = json.loads(data)
        errors = []

        if (
            "input" not in parsed_data
            or not isinstance(parsed_data.get("input"), dict)
        ):
            errors.append(
                "Error: 'input' data is missing or not a dictionary."
            )
        else:
            if "swarm_id" not in parsed_data["input"]:
                errors.append(
                    "Error: 'swarm_id' key is missing in the 'input'."
                )

            if "name" not in parsed_data["input"]:
                errors.append(
                    "Error: 'name' key is missing in the 'input'."
                )

            if "flow" not in parsed_data["input"]:
                errors.append(
                    "Error: 'flow' key is missing in the 'input'."
                )

        if "time" not in parsed_data:
            errors.append("Error: 'time' key is missing.")

        if errors:
            logger.error(f"Errors found while parsing AgentRearrange output: {errors}")
            return "\n".join(errors)

        swarm_id = parsed_data["input"]["swarm_id"]
        swarm_name = parsed_data["input"]["name"]
        agent_flow = parsed_data["input"]["flow"]
        overall_time = parsed_data["time"]

        output = f"Workflow Execution Details\n\n"
        output += f"Swarm ID: `{swarm_id}`\n"
        output += f"Swarm Name: `{swarm_name}`\n"
        output += f"Agent Flow: `{agent_flow}`\n\n---\n"
        output += f"Agent Task Execution\n\n"

        if "outputs" not in parsed_data:
            errors.append("Error: 'outputs' key is missing")
        elif parsed_data["outputs"] is None:
            errors.append("Error: 'outputs' data is None")
        elif not isinstance(parsed_data["outputs"], list):
            errors.append("Error: 'outputs' data is not a list.")
        elif not parsed_data["outputs"]:
            errors.append("Error: 'outputs' list is empty.")

        if errors:
            logger.error(f"Errors found while parsing AgentRearrange output: {errors}")
            return "\n".join(errors)

        for i, agent_output in enumerate(
            parsed_data["outputs"], start=3
        ):
            if not isinstance(agent_output, dict):
                errors.append(
                    f"Error: Agent output at index {i} is not a"
                    " dictionary"
                )
                continue

            if "agent_name" not in agent_output:
                errors.append(
                    f"Error: 'agent_name' key is missing at index {i}"
                )
                continue

            if "steps" not in agent_output:
                errors.append(
                    f"Error: 'steps' key is missing at index {i}"
                )
                continue

            if agent_output["steps"] is None:
                errors.append(
                    f"Error: 'steps' data is None at index {i}"
                )
                continue

            if not isinstance(agent_output["steps"], list):
                errors.append(
                    f"Error: 'steps' data is not a list at index {i}"
                )
                continue

            if not agent_output["steps"]:
                errors.append(
                    f"Error: 'steps' list is empty at index {i}"
                )
                continue

            agent_name = agent_output["agent_name"]
            output += f"Run {(3-i)} (Agent: `{agent_name}`)**\n\n"
            # output += "<details>\n<summary>Show/Hide Agent Steps</summary>\n\n"

            # Iterate over steps
            for j, step in enumerate(agent_output["steps"], start=3):
                if not isinstance(step, dict):
                    errors.append(
                        f"Error: step at index {j} is not a dictionary"
                        f" at {i} agent output."
                    )
                    continue

                if step is None:
                    errors.append(
                        f"Error: step at index {j} is None at {i} agent"
                        " output"
                    )
                    continue

                if "role" not in step:
                    errors.append(
                        f"Error: 'role' key missing at step {j} at {i}"
                        " agent output."
                    )
                    continue

                if "content" not in step:
                    errors.append(
                        f"Error: 'content' key missing at step {j} at"
                        f" {i} agent output."
                    )
                    continue

                if step["role"].strip() != "System:":  # Filter out system prompts
                    #  role = step["role"]
                    content = step["content"]
                    output += f"Step {(3-j)}: \n"
                    output += f"Response :\n {content}\n\n"

            # output += "</details>\n\n---\n"

        output += f"Overall Completion Time: `{overall_time}`"
        if errors:
            logger.error(f"Errors found while parsing AgentRearrange output: {errors}")
            return "\n".join(errors)
        else:
            logger.info("AgentRearrange output parsed successfully.")
            return output
    except json.JSONDecodeError as e:
        logger.error(f"Error during parsing AgentRearrange output: {e}", exc_info=True)
        return f"Error during parsing: json.JSONDecodeError {e}"

    except Exception as e:
        logger.error(f"Error during parsing AgentRearrange output: {e}", exc_info=True)
        return f"Error during parsing: {str(e)}"

def parse_mixture_of_agents_output(data: Optional[str], error_display=None) -> str:
    """Parses the MixtureOfAgents output string and formats it for display."""
    logger.info("Parsing MixtureOfAgents output...")
    if data is None:
        logger.error("No data provided for parsing MixtureOfAgents output.")
        return "Error: No data provided for parsing."
    
    print(f"Raw data received for parsing:\n{data}")  # Debug: Print raw data

    try:
        parsed_data = json.loads(data)
        
        if "InputConfig" not in parsed_data or not isinstance(parsed_data["InputConfig"], dict):
            logger.error("Error: 'InputConfig' data is missing or not a dictionary.")
            return "Error: 'InputConfig' data is missing or not a dictionary."
        
        if "name" not in parsed_data["InputConfig"]:
            logger.error("Error: 'name' key is missing in 'InputConfig'.")
            return "Error: 'name' key is missing in 'InputConfig'."
        if "description" not in parsed_data["InputConfig"]:
           logger.error("Error: 'description' key is missing in 'InputConfig'.")
           return "Error: 'description' key is missing in 'InputConfig'."
        
        if "agents" not in parsed_data["InputConfig"] or not isinstance(parsed_data["InputConfig"]["agents"], list) :
            logger.error("Error: 'agents' key is missing in 'InputConfig' or not a list.")
            return "Error: 'agents' key is missing in 'InputConfig' or not a list."
        

        name = parsed_data["InputConfig"]["name"]
        description = parsed_data["InputConfig"]["description"]

        output = f"Mixture of Agents Workflow Details\n\n"
        output += f"Name: `{name}`\n"
        output += f"Description: `{description}`\n\n---\n"
        output += f"Agent Task Execution\n\n"
        
        for agent in parsed_data["InputConfig"]["agents"]:
            if not isinstance(agent, dict):
               logger.error("Error: agent is not a dict in InputConfig agents")
               return "Error: agent is not a dict in InputConfig agents"
            if "agent_name" not in agent:
                logger.error("Error: 'agent_name' key is missing in agents.")
                return "Error: 'agent_name' key is missing in agents."
            
            if "system_prompt" not in agent:
                 logger.error("Error: 'system_prompt' key is missing in agents.")
                 return f"Error: 'system_prompt' key is missing in agents."

            agent_name = agent["agent_name"]
            # system_prompt = agent["system_prompt"]
            output += f"Agent: `{agent_name}`\n"
            # output += f"*   **System Prompt:** `{system_prompt}`\n\n"

        if "normal_agent_outputs" not in parsed_data or not isinstance(parsed_data["normal_agent_outputs"], list) :
              logger.error("Error: 'normal_agent_outputs' key is missing or not a list.")
              return "Error: 'normal_agent_outputs' key is missing or not a list."
        
        for i, agent_output in enumerate(parsed_data["normal_agent_outputs"], start=3):
             if not isinstance(agent_output, dict):
                logger.error(f"Error: agent output at index {i} is not a dictionary.")
                return f"Error: agent output at index {i} is not a dictionary."
             if "agent_name" not in agent_output:
                logger.error(f"Error: 'agent_name' key is missing at index {i}")
                return f"Error: 'agent_name' key is missing at index {i}"
             if "steps" not in agent_output:
                logger.error(f"Error: 'steps' key is missing at index {i}")
                return f"Error: 'steps' key is missing at index {i}"
             
             if agent_output["steps"] is None:
                 logger.error(f"Error: 'steps' is None at index {i}")
                 return f"Error: 'steps' is None at index {i}"
             if not isinstance(agent_output["steps"], list):
                logger.error(f"Error: 'steps' data is not a list at index {i}.")
                return f"Error: 'steps' data is not a list at index {i}."

             agent_name = agent_output["agent_name"]
             output += f"Run {(3-i)} (Agent: `{agent_name}`)\n\n"
            #  output += "<details>\n<summary>Show/Hide Agent Steps</summary>\n\n"
             for j, step in enumerate(agent_output["steps"], start=3):
                 if not isinstance(step, dict):
                     logger.error(f"Error: step at index {j} is not a dictionary at {i} agent output.")
                     return f"Error: step at index {j} is not a dictionary at {i} agent output."
                 
                 if step is None:
                     logger.error(f"Error: step at index {j} is None at {i} agent output.")
                     return f"Error: step at index {j} is None at {i} agent output."
                 
                 if "role" not in step:
                     logger.error(f"Error: 'role' key missing at step {j} at {i} agent output.")
                     return f"Error: 'role' key missing at step {j} at {i} agent output."

                 if "content" not in step:
                     logger.error(f"Error: 'content' key missing at step {j} at {i} agent output.")
                     return f"Error: 'content' key missing at step {j} at {i} agent output."
                
                 if step["role"].strip() != "System:":  # Filter out system prompts
                    #  role = step["role"]
                     content = step["content"]
                     output += f"Step {(3-j)}: \n"
                     output += f"Response:\n {content}\n\n"

            #  output += "</details>\n\n---\n"

        if "aggregator_agent_summary" in parsed_data:
            output += f"\nAggregated Summary :\n{parsed_data['aggregator_agent_summary']}\n{'=' * 50}\n"
        logger.info("MixtureOfAgents output parsed successfully.")
        return output

    except json.JSONDecodeError as e:
         logger.error(f"Error during parsing MixtureOfAgents output: {e}", exc_info=True)
         return f"Error during parsing json.JSONDecodeError : {e}"
    
    except Exception as e:
        logger.error(f"Error during parsing MixtureOfAgents output: {e}", exc_info=True)
        return f"Error during parsing: {str(e)}"

def parse_sequential_workflow_output(data: Optional[str], error_display=None) -> str:
   """Parses the SequentialWorkflow output string and formats it for display."""
   logger.info("Parsing SequentialWorkflow output...")
   if data is None:
        logger.error("No data provided for parsing SequentialWorkflow output.")
        return "Error: No data provided for parsing."
   
   print(f"Raw data received for parsing:\n{data}")  # Debug: Print raw data

   try:
        parsed_data = json.loads(data)

        if "input" not in parsed_data or not isinstance(parsed_data.get("input"), dict):
            logger.error("Error: 'input' data is missing or not a dictionary.")
            return "Error: 'input' data is missing or not a dictionary."
        
        if "swarm_id" not in parsed_data["input"] :
           logger.error("Error: 'swarm_id' key is missing in the 'input'.")
           return "Error: 'swarm_id' key is missing in the 'input'."
        
        if "name" not in parsed_data["input"]:
           logger.error("Error: 'name' key is missing in the 'input'.")
           return "Error: 'name' key is missing in the 'input'."
        
        if "flow" not in parsed_data["input"]:
            logger.error("Error: 'flow' key is missing in the 'input'.")
            return "Error: 'flow' key is missing in the 'input'."
        
        if "time" not in parsed_data :
            logger.error("Error: 'time' key is missing.")
            return "Error: 'time' key is missing."

        swarm_id = parsed_data["input"]["swarm_id"]
        swarm_name = parsed_data["input"]["name"]
        agent_flow = parsed_data["input"]["flow"]
        overall_time = parsed_data["time"]

        output = f"Workflow Execution Details\n\n"
        output += f"Swarm ID: `{swarm_id}`\n"
        output += f"Swarm Name: `{swarm_name}`\n"
        output += f"Agent Flow: `{agent_flow}`\n\n---\n"
        output += f"Agent Task Execution\n\n"

        if "outputs" not in parsed_data:
             logger.error("Error: 'outputs' key is missing")
             return "Error: 'outputs' key is missing"
            
        if  parsed_data["outputs"] is None:
            logger.error("Error: 'outputs' data is None")
            return "Error: 'outputs' data is None"
            
        if not isinstance(parsed_data["outputs"], list):
            logger.error("Error: 'outputs' data is not a list.")
            return "Error: 'outputs' data is not a list."

        for i, agent_output in enumerate(parsed_data["outputs"], start=3):
            if not isinstance(agent_output, dict):
                logger.error(f"Error: Agent output at index {i} is not a dictionary")
                return f"Error: Agent output at index {i} is not a dictionary"
            
            if "agent_name" not in agent_output:
                logger.error(f"Error: 'agent_name' key is missing at index {i}")
                return f"Error: 'agent_name' key is missing at index {i}"
            
            if "steps" not in agent_output:
                logger.error(f"Error: 'steps' key is missing at index {i}")
                return f"Error: 'steps' key is missing at index {i}"
            
            if agent_output["steps"] is None:
                logger.error(f"Error: 'steps' data is None at index {i}")
                return f"Error: 'steps' data is None at index {i}"
            
            if not isinstance(agent_output["steps"], list):
                logger.error(f"Error: 'steps' data is not a list at index {i}")
                return f"Error: 'steps' data is not a list at index {i}"

            agent_name = agent_output["agent_name"]
            output += f"Run {(3-i)} (Agent: `{agent_name}`)\n\n"
            # output += "<details>\n<summary>Show/Hide Agent Steps</summary>\n\n"

            # Iterate over steps
            for j, step in enumerate(agent_output["steps"], start=3):
                 if not isinstance(step, dict):
                     logger.error(f"Error: step at index {j} is not a dictionary at {i} agent output.")
                     return f"Error: step at index {j} is not a dictionary at {i} agent output."
                 
                 if step is None:
                     logger.error(f"Error: step at index {j} is None at {i} agent output")
                     return f"Error: step at index {j} is None at {i} agent output"

                 if "role" not in step:
                     logger.error(f"Error: 'role' key missing at step {j} at {i} agent output.")
                     return f"Error: 'role' key missing at step {j} at {i} agent output."
                 
                 if "content" not in step:
                    logger.error(f"Error: 'content' key missing at step {j} at {i} agent output.")
                    return f"Error: 'content' key missing at step {j} at {i} agent output."
                 
                 if step["role"].strip() != "System:":  # Filter out system prompts
                    #  role = step["role"]
                     content = step["content"]
                     output += f"Step {(3-j)}:\n"
                     output += f"Response : {content}\n\n"

            # output += "</details>\n\n---\n"

        output += f"Overall Completion Time: `{overall_time}`"
        logger.info("SequentialWorkflow output parsed successfully.")
        return output

   except json.JSONDecodeError as e :
        logger.error(f"Error during parsing SequentialWorkflow output: {e}", exc_info=True)
        return f"Error during parsing json.JSONDecodeError : {e}"
   
   except Exception as e:
        logger.error(f"Error during parsing SequentialWorkflow output: {e}", exc_info=True)
        return f"Error during parsing: {str(e)}"

def parse_spreadsheet_swarm_output(file_path: str, error_display=None) -> str:
    """Parses the SpreadSheetSwarm output CSV file and formats it for display."""
    logger.info("Parsing SpreadSheetSwarm output...")
    if not file_path:
        logger.error("No file path provided for parsing SpreadSheetSwarm output.")
        return "Error: No file path provided for parsing."

    print(f"Parsing spreadsheet output from: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader, None)  # Read the header row
            if not header:
                logger.error("CSV file is empty or has no header.")
                return "Error: CSV file is empty or has no header"
            
            output = "### Spreadsheet Swarm Output ###\n\n"
            output += "| " + " | ".join(header) + " |\n" # Adding header
            output += "| " + " | ".join(["---"] * len(header)) + " |\n" # Adding header seperator

            for row in csv_reader:
                output += "| " + " | ".join(row) + " |\n" # Adding row
            
            output += "\n"
        logger.info("SpreadSheetSwarm output parsed successfully.")
        return output
    
    except FileNotFoundError as e:
        logger.error(f"Error during parsing SpreadSheetSwarm output: {e}", exc_info=True)
        return "Error: CSV file not found."
    except Exception as e:
        logger.error(f"Error during parsing SpreadSheetSwarm output: {e}", exc_info=True)
        return f"Error during parsing CSV file: {str(e)}"
def parse_json_output(data:str, error_display=None) -> str:
    """Parses a JSON string and formats it for display."""
    logger.info("Parsing JSON output...")
    if not data:
        logger.error("No data provided for parsing JSON output.")
        return "Error: No data provided for parsing."

    print(f"Parsing json output from: {data}")
    try:
        parsed_data = json.loads(data)

        output = "### Swarm Metadata ###\n\n"
        
        for key,value in parsed_data.items():
            if key == "outputs":
              output += f"**{key}**:\n"
              if isinstance(value, list):
                for item in value:
                   output += f"  -  Agent Name : {item.get('agent_name', 'N/A')}\n"
                   output += f"     Task : {item.get('task', 'N/A')}\n"
                   output += f"     Result : {item.get('result', 'N/A')}\n"
                   output += f"     Timestamp : {item.get('timestamp', 'N/A')}\n\n"

              else :
                  output += f"  {value}\n"
            
            else :
                output += f"**{key}**: {value}\n"
        logger.info("JSON output parsed successfully.")
        return output
    
    except json.JSONDecodeError as e:
         logger.error(f"Error during parsing JSON output: {e}", exc_info=True)
         return f"Error: Invalid JSON format - {e}"
    
    except Exception as e:
        logger.error(f"Error during parsing JSON output: {e}", exc_info=True)
        return f"Error during JSON parsing: {str(e)}"

class UI:
    def __init__(self, theme):
        self.theme = theme
        self.blocks = gr.Blocks(theme=self.theme)
        self.components = {}  # Dictionary to store UI components

    def create_markdown(self, text, is_header=False):
        if is_header:
            markdown = gr.Markdown(
                f"<h1 style='color: #ffffff; text-align:"
                f" center;'>{text}</h1>"
            )
        else:
            markdown = gr.Markdown(
                f"<p style='color: #cccccc; text-align:"
                f" center;'>{text}</p>"
            )
        self.components[f"markdown_{text}"] = markdown
        return markdown

    def create_text_input(self, label, lines=3, placeholder=""):
        text_input = gr.Textbox(
            label=label,
            lines=lines,
            placeholder=placeholder,
            elem_classes=["custom-input"],
        )
        self.components[f"text_input_{label}"] = text_input
        return text_input

    def create_slider(
        self, label, minimum=0, maximum=1, value=0.5, step=0.1
    ):
        slider = gr.Slider(
            minimum=minimum,
            maximum=maximum,
            value=value,
            step=step,
            label=label,
            interactive=True,
        )
        self.components[f"slider_{label}"] = slider
        return slider

    def create_dropdown(
        self, label, choices, value=None, multiselect=False
    ):
        if not choices:
            choices = ["No options available"]
        if value is None and choices:
            value = choices[0] if not multiselect else [choices[0]]

        dropdown = gr.Dropdown(
            label=label,
            choices=choices,
            value=value,
            interactive=True,
            multiselect=multiselect,
        )
        self.components[f"dropdown_{label}"] = dropdown
        return dropdown

    def create_button(self, text, variant="primary"):
        button = gr.Button(text, variant=variant)
        self.components[f"button_{text}"] = button
        return button

    def create_text_output(self, label, lines=10, placeholder=""):
        text_output = gr.Textbox(
            label=label,
            interactive=False,
            placeholder=placeholder,
            lines=lines,
            elem_classes=["custom-output"],
        )
        self.components[f"text_output_{label}"] = text_output
        return text_output

    def create_tab(self, label, content_function):
        with gr.Tab(label):
            content_function(self)

    def set_event_listener(self, button, function, inputs, outputs):
        button.click(function, inputs=inputs, outputs=outputs)

    def get_components(self, *keys):
        if not keys:
            return self.components  # return all components
        return [self.components[key] for key in keys]

    def create_json_output(self, label, placeholder=""):
        json_output = gr.JSON(
            label=label,
            value={},
            elem_classes=["custom-output"],
        )
        self.components[f"json_output_{label}"] = json_output
        return json_output

    def build(self):
        return self.blocks

    def create_conditional_input(
        self, component, visible_when, watch_component
    ):
        """Create an input that's only visible under certain conditions"""
        watch_component.change(
            fn=lambda x: gr.update(visible=visible_when(x)),
            inputs=[watch_component],
            outputs=[component],
        )


    @staticmethod
    def create_ui_theme(primary_color="red"):
        import darkdetect  # First install: pip install darkdetect
        
        # Detect system theme
        is_dark = darkdetect.isDark()
        
        # Set text colors based on system theme
        text_color = "#f0f0f0" if is_dark else "#000000"
        bg_color = "#20252c" if is_dark else "#ffffff"
        
        # Enforce theme settings
        return gr.themes.Ocean(
            primary_hue=primary_color,
            secondary_hue=primary_color,
            neutral_hue="gray",
        ).set(
            body_background_fill=bg_color,
            body_text_color=text_color,
            button_primary_background_fill=primary_color,
            button_primary_text_color=text_color,
            button_secondary_background_fill=primary_color,
            button_secondary_text_color=text_color,
            shadow_drop="0px 2px 4px rgba(0, 0, 0, 0.3)",
        )


    def create_agent_details_tab(self):
        """Create the agent details tab content."""
        with gr.Column():
            gr.Markdown("### Agent Details")
            gr.Markdown(
                """
            **Available Agent Types:**
            - Data Extraction Agent: Specialized in extracting relevant information
            - Summary Agent            - Analysis Agent: Performs detailed analysis of data

            **Swarm Types:**
            - ConcurrentWorkflow: Agents work in parallel
            - SequentialWorkflow: Agents work in sequence
            - AgentRearrange: Custom agent execution flow
            - MixtureOfAgents: Combines multiple agents with an aggregator
            - SpreadSheetSwarm: Specialized for spreadsheet operations
            - Auto: Automatically determines optimal workflow

            **Note:**
            Spreasheet swarm saves data in csv, will work in local setup !
            """
            )
            return gr.Column()

    def create_logs_tab(self):
        """Create the logs tab content."""
        with gr.Column():
            gr.Markdown("### Execution Logs")
            logs_display = gr.Textbox(
                label="System Logs",
                placeholder="Execution logs will appear here...",
                interactive=False,
                lines=10,
            )
            return logs_display
def update_flow_agents(agent_keys):
    """Update flow agents based on selected agent prompts."""
    if not agent_keys:
        return [], "No agents selected"
    agent_names = [key for key in agent_keys]
    print(f"Flow agents: {agent_names}")  # Debug: Print flow agents
    return agent_names, "Select agents in execution order"

def update_flow_preview(selected_flow_agents):
    """Update flow preview based on selected agents."""
    if not selected_flow_agents:
        return "Flow will be shown here..."
    flow = " -> ".join(selected_flow_agents)
    return flow

def create_app():
    # Initialize UI
    theme = UI.create_ui_theme(primary_color="red")
    ui = UI(theme=theme)
    global AGENT_PROMPTS
    # Available providers and models
    providers = [
        "openai",
        "anthropic",
        "cohere",
        "gemini",
        "mistral",
        "groq",
        "perplexity",
    ]

    filtered_models = {}

    for provider in providers:
        filtered_models[provider] = models_by_provider.get(provider, [])

    with ui.blocks:
        with gr.Row():
            with gr.Column(scale=4):  # Left column (80% width)
                ui.create_markdown("<b>Swarms</b>")
                ui.create_markdown(
                    "<b>The Enterprise-Grade Production-Ready Multi-Agent"
                    " Orchestration Framework</b>"
                )
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Row():
                            task_input = gr.Textbox(
                                label="Task Description",
                                placeholder="Describe your task here...",
                                lines=3,
                            )
                        with gr.Row():
                            with gr.Column(scale=1):
                                with gr.Row():
                                    # Provider selection dropdown
                                    provider_dropdown = gr.Dropdown(
                                        label="Select Provider",
                                        choices=providers,
                                        value=providers[0]
                                        if providers
                                        else None,
                                        interactive=True,
                                    )
                                #  with gr.Row():
                                #     # Model selection dropdown (initially empty)
                                    model_dropdown = gr.Dropdown(
                                        label="Select Model",
                                        choices=[],
                                        interactive=True,
                                    )
                                with gr.Row():
                                    # API key input
                                    api_key_input = gr.Textbox(
                                        label="API Key",
                                        placeholder="Enter your API key",
                                        type="password",
                                    )
                            with gr.Column(scale=1):
                                with gr.Row():
                                    dynamic_slider = gr.Slider(
                                        label="Dyn. Temp",
                                        minimum=0,
                                        maximum=1,
                                        value=0.1,
                                        step=0.01,
                                    )

                                #  with gr.Row():
                                #     max tokens slider
                                    max_loops_slider = gr.Slider(
                                        label="Max Loops",
                                        minimum=1,
                                        maximum=10,
                                        value=1,
                                        step=1,
                                    )

                                with gr.Row():
                                    # max tokens slider
                                    max_tokens_slider = gr.Slider(
                                        label="Max Tokens",
                                        minimum=100,
                                        maximum=10000,
                                        value=4000,
                                        step=100,
                                    )

                    with gr.Column(scale=2, min_width=200):
                        with gr.Column(scale=1):
                            # Get available agent prompts
                            available_prompts = (
                                list(AGENT_PROMPTS.keys())
                                if AGENT_PROMPTS
                                else ["No agents available"]
                            )
                            agent_prompt_selector = gr.Dropdown(
                                label="Select Agent Prompts",
                                choices=available_prompts,
                                value=[available_prompts[0]]
                                if available_prompts
                                else None,
                                multiselect=True,
                                interactive=True,
                            )
                            # with gr.Column(scale=1):
                            # Get available swarm types
                            swarm_types = [
                                "SequentialWorkflow",
                                "ConcurrentWorkflow",
                                "AgentRearrange",
                                "MixtureOfAgents",
                                "SpreadSheetSwarm",
                                "auto",
                            ]
                            agent_selector = gr.Dropdown(
                                label="Select Swarm",
                                choices=swarm_types,
                                value=swarm_types[0],
                                multiselect=False,
                                interactive=True,
                            )

                            # Flow configuration components for AgentRearrange
                            with gr.Column(visible=False) as flow_config:
                                flow_text = gr.Textbox(
                                    label="Agent Flow Configuration",
                                    placeholder="Enter agent flow !",
                                    lines=2,
                                )
                                gr.Markdown(
                                    """
                                        **Flow Configuration Help:**
                                        - Enter agent names separated by ' -> '
                                        - Example: Agent1 -> Agent2 -> Agent3
                                        - Use exact agent names from the prompts above
                                        """
                                )
                            # Create Agent Prompt Section
                            with gr.Accordion(
                                "Create Agent Prompt", open=False
                            ) as create_prompt_accordion:
                                with gr.Row():
                                    with gr.Column():
                                        new_agent_name_input = gr.Textbox(
                                            label="New Agent Name"
                                        )
                                    with gr.Column():
                                        new_agent_prompt_input = (
                                            gr.Textbox(
                                                label="New Agent Prompt",
                                                lines=3,
                                            )
                                        )
                                with gr.Row():
                                    with gr.Column():
                                        create_agent_button = gr.Button(
                                            "Save New Prompt"
                                        )
                                    with gr.Column():
                                        create_agent_status = gr.Textbox(
                                            label="Status",
                                            interactive=False,
                                        )

                        #  with gr.Row():
                        #     temperature_slider = gr.Slider(
                        #         label="Temperature",
                        #         minimum=0,
                        #         maximum=1,
                        #         value=0.1,
                        #         step=0.01
                        #     )

                        # Hidden textbox to store API Key
                        env_api_key_textbox = gr.Textbox(
                            value="", visible=False
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        run_button = gr.Button(
                            "Run Task", variant="primary"
                        )
                        cancel_button = gr.Button(
                            "Cancel", variant="secondary"
                        )
                    with gr.Column(scale=1):
                        with gr.Row():
                            loading_status = gr.Textbox(
                                label="Status",
                                value="Ready",
                                interactive=False,
                            )

                # Add loading indicator and status
                with gr.Row():
                    agent_output_display = gr.Textbox(
                        label="Agent Responses",
                        placeholder="Responses will appear here...",
                        interactive=False,
                        lines=10,
                    )
                with gr.Row():
                    log_display = gr.Textbox(
                        label="Logs",
                        placeholder="Logs will be displayed here...",
                        interactive=False,
                        lines=5,
                        visible=False,
                    )
                    error_display = gr.Textbox(
                       label="Error",
                       placeholder="Errors will be displayed here...",
                       interactive=False,
                       lines=5,
                       visible=False,
                    )
                def update_agent_dropdown():
                    """Update agent dropdown when a new agent is added"""
                    global AGENT_PROMPTS
                    AGENT_PROMPTS = load_prompts_from_json()
                    available_prompts = (
                        list(AGENT_PROMPTS.keys())
                        if AGENT_PROMPTS
                        else ["No agents available"]
                    )
                    return gr.update(
                        choices=available_prompts,
                        value=available_prompts[0]
                        if available_prompts
                        else None,
                    )

                def update_ui_for_swarm_type(swarm_type):
                    """Update UI components based on selected swarm type."""
                    is_agent_rearrange = swarm_type == "AgentRearrange"
                    is_mixture = swarm_type == "MixtureOfAgents"
                    is_spreadsheet = swarm_type == "SpreadSheetSwarm"

                    max_loops = (
                        5 if is_mixture or is_spreadsheet else 10
                    )

                    # Return visibility state for flow configuration and max loops update
                    return (
                        gr.update(visible=is_agent_rearrange),  # For flow_config
                        gr.update(
                            maximum=max_loops
                        ),  # For max_loops_slider
                        f"Selected {swarm_type}",  # For loading_status
                    )

                def update_model_dropdown(provider):
                    """Update model dropdown based on selected provider."""
                    models = filtered_models.get(provider, [])
                    return gr.update(
                        choices=models,
                        value=models[0] if models else None,
                    )

                def save_new_agent_prompt(agent_name, agent_prompt):
                    """Saves a new agent prompt to the JSON file."""
                    try:
                        if not agent_name or not agent_prompt:
                            return (
                                "Error: Agent name and prompt cannot be"
                                " empty."
                            )

                        if (
                            not agent_name.isalnum()
                            and "_" not in agent_name
                        ):
                            return (
                                "Error : Agent name must be alphanumeric or"
                                " underscore(_) "
                            )

                        if "agent." + agent_name in AGENT_PROMPTS:
                            return "Error : Agent name already exists"

                        with open(
                            PROMPT_JSON_PATH, "r+", encoding="utf-8"
                        ) as f:
                            try:
                                data = json.load(f)
                            except json.JSONDecodeError:
                                data = {}

                            data[agent_name] = {
                                "system_prompt": agent_prompt
                            }
                            f.seek(0)
                            json.dump(data, f, indent=4)
                            f.truncate()

                        return "New agent prompt saved successfully"

                    except Exception as e:
                        return f"Error saving agent prompt {str(e)}"

                # In the run_task_wrapper function, modify the API key handling

                async def run_task_wrapper(
                    task,
                    max_loops,
                    dynamic_temp,
                    swarm_type,
                    agent_prompt_selector,
                    flow_text,
                    provider,
                    model_name,
                    api_key,
                    temperature,
                    max_tokens,
                ):
                    """Execute the task and update the UI with progress."""
                    try:
                        # Update status
                        yield "Processing...", "Running task...", "", gr.update(visible=False), gr.update(visible=False)

                        # Prepare flow for AgentRearrange
                        flow = None
                        if swarm_type == "AgentRearrange":
                            if not flow_text:
                                yield (
                                    "Please provide the agent flow"
                                    " configuration.",
                                    "Error: Flow not configured",
                                    "",
                                    gr.update(visible=True),
                                    gr.update(visible=False)
                                )
                                return
                            flow = flow_text

                        print(
                            f"Flow string: {flow}"
                        )  # Debug: Print flow string

                        # save api keys in memory
                        api_keys[provider] = api_key

                        agents = initialize_agents(
                            dynamic_temp,
                            agent_prompt_selector,
                            model_name,
                            provider,
                            api_keys.get(provider),  # Access API key from the dictionary
                            temperature,
                            max_tokens,
                        )
                        print(
                            "Agents passed to SwarmRouter:"
                            f" {[agent.agent_name for agent in agents]}"
                        )  # Debug: Print agent list

                        # Convert agent list to dictionary
                        agents_dict = {
                            agent.agent_name: agent for agent in agents
                        }

                        # Execute task
                        async for result, router, error in execute_task(
                            task=task,
                            max_loops=max_loops,
                            dynamic_temp=dynamic_temp,
                            swarm_type=swarm_type,
                            agent_keys=agent_prompt_selector,
                            flow=flow,
                            model_name=model_name,
                            provider=provider,
                            api_key=api_keys.get(provider), # Pass the api key from memory
                            temperature=temperature,
                            max_tokens=max_tokens,
                            agents=agents_dict,  # Changed here
                            log_display=log_display,
                            error_display = error_display
                        ):
                            if error:
                                yield f"Error: {error}", f"Error: {error}", "", gr.update(visible=True), gr.update(visible=True)
                                return
                            if result is not None:
                                formatted_output = format_output(result, swarm_type, error_display)
                                yield formatted_output, "Completed", api_key, gr.update(visible=False), gr.update(visible=False)
                                return
                    except Exception as e:
                        yield f"Error: {str(e)}", f"Error: {str(e)}", "", gr.update(visible=True), gr.update(visible=True)
                        return

                # Save API key to .env
                    env_path = find_dotenv()
                    if not env_path:
                        env_path = os.path.join(os.getcwd(), ".env")
                        with open(env_path, "w") as f:
                            f.write("")
                    if not env_path:
                        env_path = os.path.join(os.getcwd(), ".env")
                        with open(env_path, "w") as f:
                            f.write("")
                    if provider == "openai":
                        set_key(env_path, "OPENAI_API_KEY", api_key)
                    elif provider == "anthropic":
                        set_key(
                            env_path, "ANTHROPIC_API_KEY", api_key
                        )
                    elif provider == "cohere":
                        set_key(env_path, "COHERE_API_KEY", api_key)
                    elif provider == "gemini":
                        set_key(env_path, "GEMINI_API_KEY", api_key)
                    elif provider == "mistral":
                        set_key(env_path, "MISTRAL_API_KEY", api_key)
                    elif provider == "groq":
                        set_key(env_path, "GROQ_API_KEY", api_key)
                    elif provider == "perplexity":
                        set_key(
                            env_path, "PERPLEXITY_API_KEY", api_key
                        )
                    else:
                        yield (
                            f"Error: {provider} this provider is not"
                            " present",
                            f"Error: {provider} not supported",
                            "",
                            gr.update(visible=True),
                            gr.update(visible=False)
                        )
                        return
                    
                # Connect the update functions
                agent_selector.change(
                    fn=update_ui_for_swarm_type,
                    inputs=[agent_selector],
                    outputs=[
                        flow_config,
                        max_loops_slider,
                        loading_status,
                    ],
                )
                provider_dropdown.change(
                    fn=update_model_dropdown,
                    inputs=[provider_dropdown],
                    outputs=[model_dropdown],
                )
                # Event for creating new agent prompts
                create_agent_button.click(
                    fn=save_new_agent_prompt,
                    inputs=[new_agent_name_input, new_agent_prompt_input],
                    outputs=[create_agent_status],
                ).then(
                    fn=update_agent_dropdown,
                    inputs=None,
                    outputs=[agent_prompt_selector],
                )

                # Create event trigger
                # Create event trigger for run button
                run_event = run_button.click(
                    fn=run_task_wrapper,
                    inputs=[
                        task_input,
                        max_loops_slider,
                        dynamic_slider,
                        agent_selector,
                        agent_prompt_selector,
                        flow_text,
                        provider_dropdown,
                        model_dropdown,
                        api_key_input,
                        max_tokens_slider
                    ],
                    outputs=[
                        agent_output_display,
                        loading_status,
                        env_api_key_textbox,
                        error_display,
                        log_display,
                    ],
                )

                # Connect cancel button to interrupt processing
                def cancel_task():
                    return "Task cancelled.", "Cancelled", "", gr.update(visible=False), gr.update(visible=False)

                cancel_button.click(
                    fn=cancel_task,
                    inputs=None,
                    outputs=[
                        agent_output_display,
                        loading_status,
                        env_api_key_textbox,
                        error_display,
                        log_display
                    ],
                    cancels=run_event,
                )

            with gr.Column(scale=1):  # Right column
                with gr.Tabs():
                    with gr.Tab("Agent Details"):
                        ui.create_agent_details_tab()

                    with gr.Tab("Logs"):
                        logs_display = ui.create_logs_tab()

                        def update_logs_display():
                            """Update logs display with current logs."""
                            return ""

                        # Update logs when tab is selected
                        logs_tab = gr.Tab("Logs")
                        logs_tab.select(
                            fn=update_logs_display,
                            inputs=None,
                            outputs=[logs_display],
                        )

    return ui.build()

if __name__ == "__main__":
    app = create_app()
    app.launch()
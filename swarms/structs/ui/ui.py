import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional
import json
import time
import asyncio
import gradio as gr
from swarms.structs.agent import Agent
from swarms.structs.swarm_router import SwarmRouter
# from swarms.structs.rearrange import AgentRearrange
from swarms.utils.loguru_logger import initialize_logger
import re
import csv  # Import the csv module for csv parsing
from swarms.utils.litellm_wrapper import LiteLLM
from litellm import models_by_provider
from dotenv import set_key, find_dotenv

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

def initialize_agents(
    dynamic_temp: float,
    agent_keys: List[str],
    model_name: str,
    provider: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> List[Agent]:
    agents = []
    seen_names = set()
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
        print(
            f"Agent created: {agent.agent_name}"
        )  # Debug: Print agent name
        agents.append(agent)

    return agents

def validate_flow(flow, agents_dict):
    agent_names = flow.split("->")
    for agent in agent_names:
        agent = agent.strip()
        if agent not in agents_dict:
            raise ValueError(
                f"Agent '{agent}' specified in the flow does not exist."
            )

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
) -> Tuple[Any, "SwarmRouter", str]: # Changed the return type here
    try:
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
            return None, None, str(e)

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
                return (
                    None,
                    None,
                    "Flow configuration is required for AgentRearrange",
                )

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
            print(f"Updated Flow string: {flow}")
            router_kwargs["flow"] = flow
            router_kwargs["output_type"] = "string"  # Changed output type here
            
           

        if swarm_type == "MixtureOfAgents":
            if len(agents) < 2:
                return (
                    None,
                    None,
                    "MixtureOfAgents requires at least 2 agents",
                )

        if swarm_type == "SequentialWorkflow":
            if len(agents) < 2:
                return (
                    None,
                    None,
                    "SequentialWorkflow requires at least 2 agents",
                )

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
                #  result = await asyncio.wait_for(
                #     asyncio.to_thread(router._run, task),
                #     timeout=timeout
                # )
                #  return result, router, ""
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
                 return result, router, ""
            
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
                result = json.dumps(result)
                return result, router, ""
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(router.run, task),
                    timeout=timeout
                )
                return result, router, ""
        except asyncio.TimeoutError:
            return (
                None,
                None,
                f"Task execution timed out after {timeout} seconds",
            )
        except Exception as e:
            return None, None, str(e)

    except Exception as e:
        return None, None, str(e)

def parse_agent_rearrange_output(data: Optional[str]) -> str:
    """
    Parses the AgentRearrange output string and formats it for display.
    """
    if data is None:
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
            return "\n".join(errors)
        else:
            return output
    except json.JSONDecodeError as e:
        return f"Error during parsing: json.JSONDecodeError {e}"

    except Exception as e:
        return f"Error during parsing: {str(e)}"

def parse_mixture_of_agents_output(data: Optional[str]) -> str:
    """Parses the MixtureOfAgents output string and formats it for display."""
    if data is None:
        return "Error: No data provided for parsing."
    
    print(f"Raw data received for parsing:\n{data}")  # Debug: Print raw data

    try:
        parsed_data = json.loads(data)
        
        if "InputConfig" not in parsed_data or not isinstance(parsed_data["InputConfig"], dict):
            return "Error: 'InputConfig' data is missing or not a dictionary."
        
        if "name" not in parsed_data["InputConfig"]:
           return "Error: 'name' key is missing in 'InputConfig'."
        if "description" not in parsed_data["InputConfig"]:
           return "Error: 'description' key is missing in 'InputConfig'."
        
        if "agents" not in parsed_data["InputConfig"] or not isinstance(parsed_data["InputConfig"]["agents"], list) :
           return "Error: 'agents' key is missing in 'InputConfig' or not a list."
        

        name = parsed_data["InputConfig"]["name"]
        description = parsed_data["InputConfig"]["description"]

        output = f"Mixture of Agents Workflow Details\n\n"
        output += f"Name: `{name}`\n"
        output += f"Description: `{description}`\n\n---\n"
        output += f"Agent Task Execution\n\n"
        
        for agent in parsed_data["InputConfig"]["agents"]:
            if not isinstance(agent, dict):
               return "Error: agent is not a dict in InputConfig agents"
            if "agent_name" not in agent:
                return "Error: 'agent_name' key is missing in agents."
            
            if "system_prompt" not in agent:
                 return f"Error: 'system_prompt' key is missing in agents."

            agent_name = agent["agent_name"]
            # system_prompt = agent["system_prompt"]
            output += f"Agent: `{agent_name}`\n"
            # output += f"*   **System Prompt:** `{system_prompt}`\n\n"

        if "normal_agent_outputs" not in parsed_data or not isinstance(parsed_data["normal_agent_outputs"], list) :
              return "Error: 'normal_agent_outputs' key is missing or not a list."
        
        for i, agent_output in enumerate(parsed_data["normal_agent_outputs"], start=3):
             if not isinstance(agent_output, dict):
                return f"Error: agent output at index {i} is not a dictionary."
             if "agent_name" not in agent_output:
                 return f"Error: 'agent_name' key is missing at index {i}"
             if "steps" not in agent_output:
                return f"Error: 'steps' key is missing at index {i}"
             
             if agent_output["steps"] is None:
                 return f"Error: 'steps' is None at index {i}"
             if not isinstance(agent_output["steps"], list):
                return f"Error: 'steps' data is not a list at index {i}."

             agent_name = agent_output["agent_name"]
             output += f"Run {(3-i)} (Agent: `{agent_name}`)\n\n"
            #  output += "<details>\n<summary>Show/Hide Agent Steps</summary>\n\n"
             for j, step in enumerate(agent_output["steps"], start=3):
                 if not isinstance(step, dict):
                    return f"Error: step at index {j} is not a dictionary at {i} agent output."
                 
                 if step is None:
                     return f"Error: step at index {j} is None at {i} agent output."
                 
                 if "role" not in step:
                    return f"Error: 'role' key missing at step {j} at {i} agent output."

                 if "content" not in step:
                    return f"Error: 'content' key missing at step {j} at {i} agent output."
                
                 if step["role"].strip() != "System:":  # Filter out system prompts
                    #  role = step["role"]
                     content = step["content"]
                     output += f"Step {(3-j)}: \n"
                     output += f"Response:\n {content}\n\n"

            #  output += "</details>\n\n---\n"

        if "aggregator_agent_summary" in parsed_data:
            output += f"\nAggregated Summary :\n{parsed_data['aggregator_agent_summary']}\n{'=' * 50}\n"
        
        return output

    except json.JSONDecodeError as e:
         return f"Error during parsing json.JSONDecodeError : {e}"
    
    except Exception as e:
        return f"Error during parsing: {str(e)}"

def parse_sequential_workflow_output(data: Optional[str]) -> str:
   """Parses the SequentialWorkflow output string and formats it for display."""
   if data is None:
      return "Error: No data provided for parsing."
   
   print(f"Raw data received for parsing:\n{data}")  # Debug: Print raw data

   try:
        parsed_data = json.loads(data)

        if "input" not in parsed_data or not isinstance(parsed_data.get("input"), dict):
            return "Error: 'input' data is missing or not a dictionary."
        
        if "swarm_id" not in parsed_data["input"] :
            return "Error: 'swarm_id' key is missing in the 'input'."
        
        if "name" not in parsed_data["input"]:
           return "Error: 'name' key is missing in the 'input'."
        
        if "flow" not in parsed_data["input"]:
           return "Error: 'flow' key is missing in the 'input'."
        
        if "time" not in parsed_data :
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
            return "Error: 'outputs' key is missing"
            
        if  parsed_data["outputs"] is None:
            return "Error: 'outputs' data is None"
            
        if not isinstance(parsed_data["outputs"], list):
            return "Error: 'outputs' data is not a list."

        for i, agent_output in enumerate(parsed_data["outputs"], start=3):
            if not isinstance(agent_output, dict):
                 return f"Error: Agent output at index {i} is not a dictionary"
            
            if "agent_name" not in agent_output:
                return f"Error: 'agent_name' key is missing at index {i}"
            
            if "steps" not in agent_output:
               return f"Error: 'steps' key is missing at index {i}"
            
            if agent_output["steps"] is None:
               return f"Error: 'steps' data is None at index {i}"
            
            if not isinstance(agent_output["steps"], list):
               return f"Error: 'steps' data is not a list at index {i}"

            agent_name = agent_output["agent_name"]
            output += f"Run {(3-i)} (Agent: `{agent_name}`)\n\n"
            # output += "<details>\n<summary>Show/Hide Agent Steps</summary>\n\n"

            # Iterate over steps
            for j, step in enumerate(agent_output["steps"], start=3):
                 if not isinstance(step, dict):
                     return f"Error: step at index {j} is not a dictionary at {i} agent output."
                 
                 if step is None:
                     return f"Error: step at index {j} is None at {i} agent output"

                 if "role" not in step:
                     return f"Error: 'role' key missing at step {j} at {i} agent output."
                 
                 if "content" not in step:
                    return f"Error: 'content' key missing at step {j} at {i} agent output."
                 
                 if step["role"].strip() != "System:":  # Filter out system prompts
                    #  role = step["role"]
                     content = step["content"]
                     output += f"Step {(3-j)}:\n"
                     output += f"Response : {content}\n\n"

            # output += "</details>\n\n---\n"

        output += f"Overall Completion Time: `{overall_time}`"
        return output

   except json.JSONDecodeError as e :
        return f"Error during parsing json.JSONDecodeError : {e}"
   
   except Exception as e:
        return f"Error during parsing: {str(e)}"

def parse_spreadsheet_swarm_output(file_path: str) -> str:
    """Parses the SpreadSheetSwarm output CSV file and formats it for display."""
    if not file_path:
        return "Error: No file path provided for parsing."

    print(f"Parsing spreadsheet output from: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader, None)  # Read the header row
            if not header:
               return "Error: CSV file is empty or has no header"
            
            output = "### Spreadsheet Swarm Output ###\n\n"
            output += "| " + " | ".join(header) + " |\n" # Adding header
            output += "| " + " | ".join(["---"] * len(header)) + " |\n" # Adding header seperator

            for row in csv_reader:
                output += "| " + " | ".join(row) + " |\n" # Adding row
            
            output += "\n"
        return output
    
    except FileNotFoundError:
        return "Error: CSV file not found."
    except Exception as e:
        return f"Error during parsing CSV file: {str(e)}"
def parse_json_output(data:str) -> str:
    """Parses a JSON string and formats it for display."""
    if not data:
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

        return output
    
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {e}"
    
    except Exception as e:
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
        return gr.themes.Ocean(
            primary_hue=primary_color,
            secondary_hue=primary_color,
            neutral_hue="gray",
        ).set(
            body_background_fill="#20252c",
            body_text_color="#f0f0f0",
            button_primary_background_fill=primary_color,
            button_primary_text_color="#ffffff",
            button_secondary_background_fill=primary_color,
            button_secondary_text_color="#ffffff",
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
        "nvidia_nim",
        "huggingface",
        "perplexity",
    ]

    filtered_models = {}

    for provider in providers:
        filtered_models[provider] = models_by_provider.get(provider, [])

    with ui.blocks:
        with gr.Row():
            with gr.Column(scale=4):  # Left column (80% width)
                ui.create_markdown("Swarms", is_header=True)
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
                                        generate_agent_button = gr.Button(
                                            "Generate_Final_p"
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
                    if provider == "huggingface":
                        models = [f"huggingface/{model}" for model in models]
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
                        if not task:
                            yield (
                                "Please provide a task description.",
                                "Error: Missing task",
                                "",
                            )
                            return

                        if (
                            not agent_prompt_selector
                            or len(agent_prompt_selector) == 0
                        ):
                            yield (
                                "Please select at least one agent.",
                                "Error: No agents selected",
                                "",
                            )
                            return

                        if not provider:
                            yield (
                                "Please select a provider.",
                                "Error: No provider selected",
                                "",
                            )
                            return

                        if not model_name:
                            yield (
                                "Please select a model.",
                                "Error: No model selected",
                                "",
                            )
                            if provider == "huggingface":
                                model_name = f"huggingface/{model_name}"
                            return

                        if not api_key:
                            yield (
                                "Please enter an API Key",
                                "Error: API Key is required",
                                "",
                            )
                            return

                        # Update status
                        yield "Processing...", "Running task...", ""

                        # Prepare flow for AgentRearrange
                        flow = None
                        if swarm_type == "AgentRearrange":
                            if not flow_text:
                                yield (
                                    "Please provide the agent flow"
                                    " configuration.",
                                    "Error: Flow not configured",
                                    "",
                                )
                                return
                            flow = flow_text

                        print(
                            f"Flow string: {flow}"
                        )  # Debug: Print flow string

                        # Save API key to .env
                        env_path = find_dotenv()
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
                        elif provider == "nvidia_nim":
                            set_key(
                                env_path, "NVIDIA_NIM_API_KEY", api_key
                            )
                        elif provider == "huggingface":
                            set_key(
                                env_path, "HUGGINGFACE_API_KEY", api_key
                            )
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
                            )
                            return

                        agents = initialize_agents(
                            dynamic_temp,
                            agent_prompt_selector,
                            model_name,
                            provider,
                            api_key,
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
                        result, router, error = await execute_task(
                            task=task,
                            max_loops=max_loops,
                            dynamic_temp=dynamic_temp,
                            swarm_type=swarm_type,
                            agent_keys=agent_prompt_selector,
                            flow=flow,
                            model_name=model_name,
                            provider=provider,
                            api_key=api_key,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            agents=agents_dict,  # Changed here
                        )

                        if error:
                            yield f"Error: {error}", f"Error: {error}", ""
                            return
                        
                        if result is None:
                            yield "Error: No output from swarm.", f"Error: {error}", ""
                            return

                        # Format output based on swarm type
                        output_lines = []
                        if swarm_type == "SpreadSheetSwarm":
                            # Check if the result is a file path or JSON
                            if os.path.exists(result):
                                parsed_output = (
                                    parse_spreadsheet_swarm_output(result)
                                )
                                output_lines.append(parsed_output)
                            else:
                                parsed_output = parse_json_output(result)
                                output_lines.append(parsed_output)

                        elif swarm_type == "AgentRearrange":
                            try:
                                parsed_output = (
                                    parse_agent_rearrange_output(result)
                                )
                            except ValueError as e:
                                parsed_output = (
                                    f"Error during parsing: {e}"
                                )  # Handle ValueError
                            output_lines.append(parsed_output)
                        elif swarm_type == "MixtureOfAgents":
                            parsed_output = (
                                parse_mixture_of_agents_output(result)
                            )
                            output_lines.append(parsed_output)
                        elif swarm_type == "SequentialWorkflow":
                            parsed_output = (
                                parse_sequential_workflow_output(result)
                            )
                            output_lines.append(parsed_output)
                        elif isinstance(
                            result, dict
                        ):  # checking if result is dict or string.
                            if swarm_type == "MixtureOfAgents":
                                # Add individual agent outputs
                                for key, value in result.items():
                                    if key != "Aggregated Summary":
                                        output_lines.append(
                                            f"### {key} ###\n{value}\n"
                                        )
                                # Add aggregated summary at the end
                                if "Aggregated Summary" in result:
                                    output_lines.append(
                                        "\n### Aggregated Summary"
                                        f" ###\n{result['Aggregated Summary']}\n{'=' * 50}\n"
                                    )
                            else:  # SequentialWorkflow, ConcurrentWorkflow, Auto
                                for key, value in result.items():
                                    output_lines.append(
                                        f"### {key} ###\n{value}\n{'=' * 50}\n"
                                    )
                        elif isinstance(result, str):
                            output_lines.append(str(result))

                        yield "\n".join(output_lines), "Completed", api_key

                    except Exception as e:
                        yield f"Error: {str(e)}", f"Error: {str(e)}", ""

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
                    ],
                )

                # Connect cancel button to interrupt processing
                def cancel_task():
                    return "Task cancelled.", "Cancelled", ""

                cancel_button.click(
                    fn=cancel_task,
                    inputs=None,
                    outputs=[
                        agent_output_display,
                        loading_status,
                        env_api_key_textbox,
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
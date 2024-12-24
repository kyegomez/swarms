import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional
import json
import time
import asyncio
import gradio as gr
from swarms.structs.agent import Agent
from swarms.structs.swarm_router import SwarmRouter
from swarms.utils.loguru_logger import initialize_logger
from swarm_models import OpenAIChat


# Initialize logger
logger = initialize_logger(log_folder="swarm_ui")


# Load environment variables
load_dotenv()

# Get the OpenAI API key from the environment variable
api_key = os.getenv("GROQ_API_KEY")

# changed to swarms_models
# adding functionality to view other models of swarms models

# Model initialization
model = OpenAIChat(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=api_key,
    model_name="llama-3.1-70b-versatile",
    temperature=0.1,
)

# Define the path to agent_prompts.json

# locates the json file and then fetches the promopts

PROMPT_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_prompts.json")
logger.info(f"Loading prompts from: {PROMPT_JSON_PATH}")


def load_prompts_from_json() -> Dict[str, str]:
    try:
        if not os.path.exists(PROMPT_JSON_PATH):
            # Load default prompts
            return {
                "agent.data_extractor": "You are a data extraction agent...",
                "agent.summarizer": "You are a summarization agent...",
                "agent.onboarding_agent": "You are an onboarding agent..."
            }

        with open(PROMPT_JSON_PATH, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # Load default prompts
                return {
                    "agent.data_extractor": "You are a data extraction agent...",
                    "agent.summarizer": "You are a summarization agent...",
                    "agent.onboarding_agent": "You are an onboarding agent..."
                }

            if not isinstance(data, dict):
                # Load default prompts
                 return {
                    "agent.data_extractor": "You are a data extraction agent...",
                    "agent.summarizer": "You are a summarization agent...",
                    "agent.onboarding_agent": "You are an onboarding agent..."
                }


            prompts = {}
            for agent_name, details in data.items():
                if not isinstance(details, dict) or "system_prompt" not in details:
                   continue

                prompts[f"agent.{agent_name}"] = details["system_prompt"]

            if not prompts:
               # Load default prompts
                return {
                    "agent.data_extractor": "You are a data extraction agent...",
                    "agent.summarizer": "You are a summarization agent...",
                    "agent.onboarding_agent": "You are an onboarding agent..."
                }

            return prompts

    except Exception:
        # Load default prompts
        return {
            "agent.data_extractor": "You are a data extraction agent...",
            "agent.summarizer": "You are a summarization agent...",
            "agent.onboarding_agent": "You are an onboarding agent..."
        }


AGENT_PROMPTS = load_prompts_from_json()


def initialize_agents(dynamic_temp: float, agent_keys: List[str]) -> List[Agent]:
    agents = []
    seen_names = set()
    for agent_key in agent_keys:
        if agent_key not in AGENT_PROMPTS:
            raise ValueError(f"Invalid agent key: {agent_key}")

        agent_prompt = AGENT_PROMPTS[agent_key]
        agent_name = agent_key.split('.')[-1]

        # Ensure unique agent names
        base_name = agent_name
        counter = 1
        while agent_name in seen_names:
            agent_name = f"{base_name}_{counter}"
            counter += 1
        seen_names.add(agent_name)

        agent = Agent(
            agent_name=f"Agent-{agent_name}",
            system_prompt=agent_prompt,
            llm=model,
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
        agents.append(agent)

    return agents


async def execute_task(task: str, max_loops: int, dynamic_temp: float,
                        swarm_type: str, agent_keys: List[str], flow: str = None) -> Tuple[Any, 'SwarmRouter', str]:
    """
    Enhanced task execution with comprehensive error handling and raw result return.
    """
    try:
        # Initialize agents
        try:
            agents = initialize_agents(dynamic_temp, agent_keys)
        except Exception as e:
            return None, None, str(e)

        # Swarm-specific configurations
        router_kwargs = {
            "name": "multi-agent-workflow",
            "description": f"Executing {swarm_type} workflow",
            "max_loops": max_loops,
            "agents": agents,
            "autosave": True,
            "return_json": True,
            "output_type": "string",
            "swarm_type": swarm_type,  # Pass swarm_type here
        }

        if swarm_type == "AgentRearrange":
            if not flow:
                return None, None, "Flow configuration is required for AgentRearrange"
            router_kwargs["flow"] = flow

        if swarm_type == "MixtureOfAgents":
            if len(agents) < 2:
                return None, None, "MixtureOfAgents requires at least 2 agents"

        if swarm_type == "SpreadSheetSwarm":
            # spread sheet swarm needs specific setup
            output_dir = "swarm_outputs"
            os.makedirs(output_dir, exist_ok=True)

            # Create a sanitized filename using only safe characters
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"swarm_output_{timestamp}.csv"
            output_path = os.path.join(output_dir, output_file)

            # Validate the output path
            try:
                # Ensure the path is valid and writable
                with open(output_path, 'w') as f:
                    pass
                os.remove(output_path)  # Clean up the test file
            except OSError as e:
                return None, None, str(e)

            router_kwargs["output_path"] = output_path
        # Create and execute SwarmRouter
        try:
            timeout = 450 if swarm_type != "SpreadSheetSwarm" else 900  # SpreadSheetSwarm will have different timeout.
            router = SwarmRouter(**router_kwargs)
            
            if swarm_type == "AgentRearrange":
                 result = await asyncio.wait_for(
                    asyncio.to_thread(router._run, task),
                    timeout=timeout
                )
                 return result, router, ""
            
            result = await asyncio.wait_for(
                asyncio.to_thread(router.run, task),
                timeout=timeout
            )


            if swarm_type == "SpreadSheetSwarm":
                # Verify the output file was created
                if not os.path.exists(output_path):
                    return None, None, "Output file was not created"

                return output_path, router, ""

            return result, router, ""

        except asyncio.TimeoutError:
            return None, None, f"Task execution timed out after {timeout} seconds"
        except Exception as e:
            return None, None, str(e)

    except Exception as e:
        return None, None, str(e)


class UI:
    def __init__(self, theme):
        self.theme = theme
        self.blocks = gr.Blocks(theme=self.theme)
        self.components = {}  # Dictionary to store UI components

    def create_markdown(self, text, is_header=False):
        if is_header:
            markdown = gr.Markdown(f"<h1 style='color: #ffffff; text-align: center;'>{text}</h1>")
        else:
            markdown = gr.Markdown(f"<p style='color: #cccccc; text-align: center;'>{text}</p>")
        self.components[f'markdown_{text}'] = markdown
        return markdown

    def create_text_input(self, label, lines=3, placeholder=""):
        text_input = gr.Textbox(
            label=label,
            lines=lines,
            placeholder=placeholder,
            elem_classes=["custom-input"],
        )
        self.components[f'text_input_{label}'] = text_input
        return text_input

    def create_slider(self, label, minimum=0, maximum=1, value=0.5, step=0.1):
        slider = gr.Slider(
            minimum=minimum,
            maximum=maximum,
            value=value,
            step=step,
            label=label,
            interactive=True,
        )
        self.components[f'slider_{label}']
        return slider

    def create_dropdown(self, label, choices, value=None, multiselect=False):
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
        self.components[f'dropdown_{label}'] = dropdown
        return dropdown

    def create_button(self, text, variant="primary"):
        button = gr.Button(text, variant=variant)
        self.components[f'button_{text}'] = button
        return button

    def create_text_output(self, label, lines=10, placeholder=""):
        text_output = gr.Textbox(
            label=label,
            interactive=False,
            placeholder=placeholder,
            lines=lines,
            elem_classes=["custom-output"],
        )
        self.components[f'text_output_{label}'] = text_output
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
        self.components[f'json_output_{label}'] = json_output
        return json_output

    def build(self):
        return self.blocks

    def create_conditional_input(self, component, visible_when, watch_component):
        """Create an input that's only visible under certain conditions"""
        watch_component.change(
            fn=lambda x: gr.update(visible=visible_when(x)),
            inputs=[watch_component],
            outputs=[component]
        )

    @staticmethod
    def create_ui_theme(primary_color="red"):
        return gr.themes.Soft(
            primary_hue=primary_color,
            secondary_hue="gray",
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
            gr.Markdown("""
            **Available Agent Types:**
            - Data Extraction Agent: Specialized in extracting relevant information
            - Summary Agent: Creates concise summaries of information
            - Analysis Agent: Performs detailed analysis of data
            
            **Swarm Types:**
            - ConcurrentWorkflow: Agents work in parallel
            - SequentialWorkflow: Agents work in sequence
            - AgentRearrange: Custom agent execution flow
            - MixtureOfAgents: Combines multiple agents with an aggregator
            - SpreadSheetSwarm: Specialized for spreadsheet operations
            - Auto: Automatically determines optimal workflow
            """)
            return gr.Column()

    def create_logs_tab(self):
        """Create the logs tab content."""
        with gr.Column():
            gr.Markdown("### Execution Logs")
            logs_display = gr.Textbox(
                label="System Logs",
                placeholder="Execution logs will appear here...",
                interactive=False,
                lines=10
            )
            return logs_display


def create_app():
    # Initialize UI
    theme = UI.create_ui_theme(primary_color="red")
    ui = UI(theme=theme)

    with ui.blocks:
        with gr.Row():
            with gr.Column(scale=4):  # Left column (80% width)
                ui.create_markdown("Swarms", is_header=True)
                ui.create_markdown(
                    "<b>The Enterprise-Grade Production-Ready Multi-Agent Orchestration Framework</b>"
                )
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Row():
                            task_input = gr.Textbox(
                                label="Task Description",
                                placeholder="Describe your task here...",
                                lines=3
                            )
                        with gr.Row():
                            with gr.Column(scale=1):
                                # Get available agent prompts
                                available_prompts = list(AGENT_PROMPTS.keys()) if AGENT_PROMPTS else ["No agents available"]
                                agent_prompt_selector = gr.Dropdown(
                                    label="Select Agent Prompts",
                                    choices=available_prompts,
                                    value=[available_prompts[0]] if available_prompts else None,
                                    multiselect=True,
                                    interactive=True
                                )
                            with gr.Column(scale=1):
                                # Get available swarm types
                                swarm_types = [
                                    "SequentialWorkflow", "ConcurrentWorkflow", "AgentRearrange",
                                    "MixtureOfAgents", "SpreadSheetSwarm", "auto"
                                ]
                                agent_selector = gr.Dropdown(
                                    label="Select Swarm",
                                    choices=swarm_types,
                                    value=swarm_types[0],
                                    multiselect=False,
                                    interactive=True
                                )

                                # Flow configuration components for AgentRearrange
                                with gr.Column(visible=False) as flow_config:
                                    flow_text = gr.Textbox(
                                        label="Agent Flow Configuration",
                                        placeholder="Enter agent flow (e.g., Agent1 -> Agent2 -> Agent3)",
                                        lines=2
                                    )
                                    gr.Markdown(
                                        """
                                        **Flow Configuration Help:**
                                        - Enter agent names separated by ' -> '
                                        - Example: Agent1 -> Agent2 -> Agent3
                                        - Use exact agent names from the prompts above
                                        """
                                    )

                    with gr.Column(scale=2, min_width=200):
                        with gr.Row():
                            max_loops_slider = gr.Slider(
                                label="Max Loops",
                                minimum=1,
                                maximum=10,
                                value=1,
                                step=1
                            )

                        with gr.Row():
                            dynamic_slider = gr.Slider(
                                label="Dynamic Temp",
                                minimum=0,
                                maximum=1,
                                value=0.1,
                                step=0.01
                            )
                        with gr.Row():
                            loading_status = gr.Textbox(
                                label="Status",
                                value="Ready",
                                interactive=False
                            )

                with gr.Row():
                    run_button = gr.Button("Run Task", variant="primary")
                    cancel_button = gr.Button("Cancel", variant="secondary")

                # Add loading indicator and status
                with gr.Row():
                    agent_output_display = gr.Textbox(
                        label="Agent Responses",
                        placeholder="Responses will appear here...",
                        interactive=False,
                        lines=10
                    )

                def update_flow_agents(agent_keys):
                    """Update flow agents based on selected agent prompts."""
                    if not agent_keys:
                       return [], "No agents selected"
                    agent_names = [key.split('.')[-1] for key in agent_keys]
                    return agent_names, "Select agents in execution order"

                def update_flow_preview(selected_flow_agents):
                    """Update flow preview based on selected agents."""
                    if not selected_flow_agents:
                        return "Flow will be shown here..."
                    flow = " -> ".join(selected_flow_agents)
                    return flow

                def update_ui_for_swarm_type(swarm_type):
                    """Update UI components based on selected swarm type."""
                    is_agent_rearrange = swarm_type == "AgentRearrange"
                    is_mixture = swarm_type == "MixtureOfAgents"
                    is_spreadsheet = swarm_type == "SpreadSheetSwarm"

                    max_loops = 5 if is_mixture or is_spreadsheet else 10

                    # Return visibility state for flow configuration and max loops update
                    return (
                        gr.update(visible=is_agent_rearrange),  # For flow_config
                        gr.update(maximum=max_loops),  # For max_loops_slider
                        f"Selected {swarm_type}"  # For loading_status
                    )

                async def run_task_wrapper(task, max_loops, dynamic_temp, swarm_type, agent_prompt_selector, flow_text):
                    """Execute the task and update the UI with progress."""
                    try:
                        if not task:
                            yield "Please provide a task description.", "Error: Missing task"
                            return

                        if not agent_prompt_selector or len(agent_prompt_selector) == 0:
                            yield "Please select at least one agent.", "Error: No agents selected"
                            return


                        # Update status
                        yield "Processing...", "Running task..."

                        # Prepare flow for AgentRearrange
                        flow = None
                        if swarm_type == "AgentRearrange":
                            if not flow_text:
                                yield "Please provide the agent flow configuration.", "Error: Flow not configured"
                                return
                            flow = flow_text

                        # Execute task
                        result, router, error = await execute_task(
                            task=task,
                            max_loops=max_loops,
                            dynamic_temp=dynamic_temp,
                            swarm_type=swarm_type,
                            agent_keys=agent_prompt_selector,
                            flow=flow
                        )

                        if error:
                            yield f"Error: {error}", "Error occurred"
                            return

                        # Format output based on swarm type
                        output_lines = []
                        if swarm_type == "SpreadSheetSwarm":
                            output_lines.append(f"### Spreadsheet Output ###\n{result}\n{'=' * 50}\n")
                        elif isinstance(result, dict):  # checking if result is dict or string.
                            if swarm_type == "AgentRearrange":
                                for key, value in result.items():
                                    output_lines.append(f"### Step {key} ###\n{value}\n{'=' * 50}\n")
                            elif swarm_type == "MixtureOfAgents":
                                # Add individual agent outputs
                                for key, value in result.items():
                                    if key != "Aggregated Summary":
                                        output_lines.append(f"### {key} ###\n{value}\n")
                                # Add aggregated summary at the end
                                if "Aggregated Summary" in result:
                                    output_lines.append(f"\n### Aggregated Summary ###\n{result['Aggregated Summary']}\n{'=' * 50}\n")
                            else:  # SequentialWorkflow, ConcurrentWorkflow, Auto
                                for key, value in result.items():
                                    output_lines.append(f"### {key} ###\n{value}\n{'=' * 50}\n")
                        elif isinstance(result, str):
                            output_lines.append(str(result))

                        yield "\n".join(output_lines), "Completed"

                    except Exception as e:
                        yield f"Error: {str(e)}", "Error occurred"

                # Connect the update functions
                agent_selector.change(
                    fn=update_ui_for_swarm_type,
                    inputs=[agent_selector],
                    outputs=[flow_config, max_loops_slider, loading_status]
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
                        flow_text
                    ],
                    outputs=[agent_output_display, loading_status]
                )

                # Connect cancel button to interrupt processing
                def cancel_task():
                    return "Task cancelled.", "Cancelled"

                cancel_button.click(
                    fn=cancel_task,
                    inputs=None,
                    outputs=[agent_output_display, loading_status],
                    cancels=run_event
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
                        logs_tab.select(fn=update_logs_display, inputs=None, outputs=[logs_display])

    return ui.build()

if __name__ == "__main__":
    app = create_app()
    app.launch()
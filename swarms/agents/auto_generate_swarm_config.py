import os
import re
from typing import Optional

import yaml
from tenacity import retry, stop_after_attempt, wait_exponential

from swarms import Agent
from swarms.utils.formatter import formatter
from swarms.utils.litellm_wrapper import LiteLLM


def prepare_yaml_for_parsing(raw_yaml: str) -> str:
    """
    Prepares raw YAML content by fixing spacing and formatting issues.

    Args:
        raw_yaml (str): The raw YAML content extracted from Markdown.

    Returns:
        str: The cleaned YAML content ready for parsing.
    """
    # Fix sequence items that are improperly placed on the same line as their key
    fixed_yaml = re.sub(
        r"(\b\w+\b):\s*-\s*", r"\1:\n  - ", raw_yaml
    )  # Fix "key: - value" to "key:\n  - value"

    # Ensure proper spacing after colons
    fixed_yaml = re.sub(
        r"(\S):(\S)", r"\1: \2", fixed_yaml
    )  # Ensure space after colons

    # Remove trailing spaces before newlines
    fixed_yaml = re.sub(r"\s+\n", "\n", fixed_yaml)

    # Replace non-breaking spaces (if any) with regular spaces
    fixed_yaml = fixed_yaml.replace("\xa0", " ")

    return fixed_yaml.strip()


def parse_yaml_from_swarm_markdown(markdown_text: str) -> dict:
    """
    Extracts and prepares YAML content from a Markdown-style 'Auto-Swarm-Builder' block and parses it.

    Args:
        markdown_text (str): The Markdown text containing the YAML inside 'Auto-Swarm-Builder' block.

    Returns:
        dict: A parsed Python dictionary of the YAML content.
    """
    # Match YAML blocks inside triple backticks — use the last match
    # because the LLM's generated YAML comes after any examples in the prompt
    pattern = r"```yaml\s*\n(.*?)```"
    matches = re.findall(pattern, markdown_text, re.DOTALL)

    if not matches:
        raise ValueError(
            "No YAML content found in the 'Auto-Swarm-Builder' block."
        )

    raw_yaml = matches[-1].strip()

    # Preprocess and normalize the YAML content
    normalized_yaml = prepare_yaml_for_parsing(raw_yaml)

    return normalized_yaml


AUTO_GEN_PROMPT = """
You are a specialized agent responsible for creating YAML configuration files for multi-agent swarms. Your role is to generate well-structured YAML that defines both individual agents and swarm architectures based on user requirements.
Output only the yaml nothing else. You will be penalized for making mistakes

GUIDELINES:
1. Each YAML file must contain an `agents` section with at least one agent configuration
2. Each agent configuration requires the following mandatory fields:
   - agent_name (string)
   - system_prompt (string)

3. Optional agent fields include:
   - max_loops (integer)
   - autosave (boolean)
   - dashboard (boolean)
   - verbose (boolean)
   - dynamic_temperature_enabled (boolean)
   - saved_state_path (string)
   - user_name (string)
   - retry_attempts (integer)
   - context_length (integer)
   - return_step_meta (boolean)
   - output_type (string)
   - task (string)

4. When a swarm is needed, include a `swarm_architecture` section with:
   Mandatory fields:
   - name (string)
   - swarm_type (string: "ConcurrentWorkflow" or "SequentialWorkflow") [AgentRearrange, MixtureOfAgents, SpreadSheetSwarm, SequentialWorkflow, ConcurrentWorkflow]	
   
   Optional fields:
   - description (string)
   - max_loops (integer)
   - task (string)

TEMPLATE STRUCTURE:
```yaml
agents:
  - agent_name: "Agent-1-Name"
    system_prompt: "Detailed system prompt here"
    max_loops: 1
    # [additional optional fields]

  - agent_name: "Agent-2-Name"
    system_prompt: "Detailed system prompt here"
    # [additional optional fields]

swarm_architecture:
  name: "Swarm-Name"
  description: "Swarm purpose and goals"
  swarm_type: "ConcurrentWorkflow"
  max_loops: 5
  task: "Main swarm task description"
```

VALIDATION RULES:
1. All agent names must be unique
2. System prompts must be clear and specific to the agent's role
3. Integer values must be positive
4. Boolean values must be true or false (lowercase)
5. File paths should use forward slashes
6. Tasks should be specific and aligned with the agent/swarm purpose

When generating a YAML configuration:
1. Ask for specific requirements about the agents and swarm needed
2. Determine if a swarm architecture is necessary based on the task complexity
3. Generate appropriate system prompts for each agent based on their roles
4. Include relevant optional fields based on the use case
5. Validate the configuration against all rules before returning

Example valid YAML configurations are provided below. Use these as references for structure and formatting:

```yaml


agents:
  - agent_name: "Data-Analysis-Agent"
    system_prompt: "You are a specialized data analysis agent focused on processing and interpreting financial data. Provide clear, actionable insights based on the data provided."
    max_loops: 3
    autosave: true
    verbose: true
    context_length: 100000
    output_type: "json"
    task: "Analyze quarterly financial reports and identify trends"

# Multi-Agent Swarm Example
agents:
  - agent_name: "Research-Agent"
    system_prompt: "You are a research agent specialized in gathering and summarizing scientific publications. Focus on peer-reviewed sources and provide comprehensive summaries."
    max_loops: 2
    context_length: 150000
    output_type: "str"

  - agent_name: "Analysis-Agent"
    system_prompt: "You are an analysis agent that processes research summaries and identifies key patterns and insights. Provide detailed analytical reports."
    max_loops: 3
    context_length: 200000
    output_type: "json"

swarm_architecture:
  name: "Research-Analysis-Swarm"
  description: "A swarm for comprehensive research analysis and insight generation"
  swarm_type: "SequentialWorkflow"
  max_loops: 5
  task: "Research and analyze recent developments in quantum computing"
  
```
"""


def _slugify(text: str) -> str:
    """Convert text to a filename-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    return slug


# Agent YAML keys that map directly to Agent() constructor params
_AGENT_PARAM_MAP = {
    "agent_name": "agent_name",
    "system_prompt": "system_prompt",
    "description": "agent_description",
    "model_name": "model_name",
    "max_loops": "max_loops",
    "temperature": "temperature",
    "max_tokens": "max_tokens",
    "autosave": "autosave",
    "verbose": "verbose",
    "dynamic_temperature_enabled": "dynamic_temperature_enabled",
    "context_length": "context_length",
    "output_type": "output_type",
    "saved_state_path": "saved_state_path",
    "user_name": "user_name",
    "retry_attempts": "retry_attempts",
    "return_step_meta": "return_step_meta",
    "dashboard": "dashboard",
    "auto_generate_prompt": "auto_generate_prompt",
    "artifacts_on": "artifacts_on",
    "artifacts_file_extension": "artifacts_file_extension",
    "artifacts_output_path": "artifacts_output_path",
}

# Keys to skip when generating agent code (not Agent() constructor params)
_AGENT_SKIP_KEYS = {"task"}


def _format_value(value) -> str:
    """Format a Python value as a source-code literal."""
    if isinstance(value, str):
        # Use triple-quoted string for multi-line or strings with quotes
        if "\n" in value or ('"' in value and "'" in value):
            escaped = value.replace("\\", "\\\\").replace(
                '"""', '\\"\\"\\"'
            )
            return f'"""{escaped}"""'
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    return repr(value)


def _agent_var_name(agent_name: str) -> str:
    """Convert an agent name to a valid Python variable name."""
    var = re.sub(r"[^\w]", "_", agent_name.lower())
    var = re.sub(r"_+", "_", var).strip("_")
    if var and var[0].isdigit():
        var = f"agent_{var}"
    return var or "agent"


def _render_agent_code(agent_dict: dict) -> tuple:
    """Render a single agent dict as Python source code.

    Returns:
        (var_name, code_string)
    """
    name = agent_dict.get("agent_name", "Agent")
    var = _agent_var_name(name)

    lines = [f"{var} = Agent("]
    for yaml_key, param_name in _AGENT_PARAM_MAP.items():
        if yaml_key in agent_dict:
            val = _format_value(agent_dict[yaml_key])
            lines.append(f"    {param_name}={val},")

    # Emit unknown keys as comments
    known_keys = set(_AGENT_PARAM_MAP.keys()) | _AGENT_SKIP_KEYS
    for key in agent_dict:
        if key not in known_keys:
            lines.append(
                f"    # Unknown YAML key: {key}={agent_dict[key]!r}"
            )

    lines.append(")")
    return var, "\n".join(lines)


def write_autoswarm_file(
    config: dict,
    task: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Render a parsed swarm config dict as a ready-to-run Python file.

    Args:
        config: Parsed YAML config dict with 'agents' and optional 'swarm_architecture'.
        task: The original task string.
        output_path: Optional file path. Auto-generated from swarm name if not provided.
        output_dir: Optional directory to create the file in. Used when output_path
            is not provided. The directory is created if it does not exist.

    Returns:
        The resolved file path that was written.
    """
    agents_list = config.get("agents", [])
    swarm_arch = config.get("swarm_architecture", {})

    # Determine output path
    if not output_path:
        swarm_name = swarm_arch.get("name", "output")
        filename = f"autoswarm_{_slugify(swarm_name)}.py"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
        else:
            output_path = filename

    # Render agent blocks
    agent_vars = []
    agent_blocks = []
    for agent_dict in agents_list:
        var, code = _render_agent_code(agent_dict)
        # Deduplicate variable names
        original_var = var
        counter = 2
        while var in agent_vars:
            var = f"{original_var}_{counter}"
            code = code.replace(
                f"{original_var} = Agent(",
                f"{var} = Agent(",
                1,
            )
            counter += 1
        agent_vars.append(var)
        agent_blocks.append(code)

    # Build the file
    parts = [
        "from swarms import Agent",
        "from swarms.structs.swarm_router import SwarmRouter",
        "",
        "# Auto-generated by `swarms autoswarm` -- edit freely",
        "",
    ]

    # Agent definitions
    for block in agent_blocks:
        parts.append(block)
        parts.append("")

    # SwarmRouter definition
    if swarm_arch:
        router_lines = ["swarm = SwarmRouter("]
        if swarm_arch.get("name"):
            router_lines.append(
                f"    name={_format_value(swarm_arch['name'])},"
            )
        if swarm_arch.get("description"):
            router_lines.append(
                f"    description={_format_value(swarm_arch['description'])},"
            )
        router_lines.append(f"    agents=[{', '.join(agent_vars)}],")
        if swarm_arch.get("swarm_type"):
            router_lines.append(
                f"    swarm_type={_format_value(swarm_arch['swarm_type'])},"
            )
        if swarm_arch.get("max_loops"):
            router_lines.append(
                f"    max_loops={swarm_arch['max_loops']},"
            )
        router_lines.append(")")
        parts.append("\n".join(router_lines))
    else:
        # No swarm architecture - just create a basic sequential router
        parts.append(
            f"swarm = SwarmRouter(\n"
            f"    name='autoswarm',\n"
            f"    agents=[{', '.join(agent_vars)}],\n"
            f"    swarm_type='SequentialWorkflow',\n"
            f")"
        )

    parts.append("")

    # Main block
    parts.append('if __name__ == "__main__":')
    parts.append(f"    result = swarm.run({_format_value(task)})")
    parts.append("    print(result)")
    parts.append("")

    source = "\n".join(parts)

    with open(output_path, "w") as f:
        f.write(source)

    return os.path.abspath(output_path)


def generate_swarm_config(
    task: str,
    file_name: str = "swarm_config_output.yaml",
    model_name: str = "gpt-4.1",
    *args,
    **kwargs,
):
    """
    Generates a swarm configuration based on the provided task and model name.

    This function uses an LLM agent to produce a YAML config, parses it,
    creates agents from it, and runs the swarm. It also returns the parsed
    config dict so callers can use it (e.g. to write a Python file).

    Args:
        task (str): The task to be performed by the swarm.
        file_name (str, optional): The file name for the output YAML configuration. Defaults to "swarm_config_output.yaml".
        model_name (str, optional): The name of the model to use for the agent. Defaults to "gpt-4.1".
        *args: Additional positional arguments to be passed to the agent's run method.
        **kwargs: Additional keyword arguments to be passed to the agent's run method.

    Returns:
        dict: The parsed YAML config dict with 'agents' and 'swarm_architecture' keys.
    """
    formatter.print_panel(
        "Auto Generating Swarm...", "Auto Swarm Builder"
    )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=4, max=10),
    )
    def attempt_generate_swarm_config():
        try:
            model = LiteLLM(model_name=model_name)

            # Initialize the agent
            agent = Agent(
                agent_name="Auto-Swarm-Builder",
                system_prompt=AUTO_GEN_PROMPT,
                llm=model,
                max_loops=1,
                saved_state_path="swarm_builder.json",
                user_name="swarms_corp",
                output_type="str",
            )

            # Generate output from the agent
            raw_output = agent.run(task, *args, **kwargs)
            yaml_content = parse_yaml_from_swarm_markdown(raw_output)
            print(yaml_content)

            # Parse the YAML to get the config dict
            config_dict = yaml.safe_load(yaml_content)

            formatter.print_panel(
                "Swarm configuration generated successfully.",
                "Success",
            )

            return config_dict

        except Exception as e:
            formatter.print_panel(
                f"Error generating swarm configuration: {str(e)}",
                "Error",
            )
            raise

    return attempt_generate_swarm_config()

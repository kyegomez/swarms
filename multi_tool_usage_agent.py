from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import json
from datetime import datetime
import inspect
import typing
from typing import Union
from swarms import Agent
from swarm_models import OpenAIChat
from dotenv import load_dotenv


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str]
    callable: Optional[Callable] = None


@dataclass
class ExecutionStep:
    step_id: str
    tool_name: str
    parameters: Dict[str, Any]
    purpose: str
    depends_on: List[str]
    completed: bool = False
    result: Optional[Any] = None


def extract_type_hints(func: Callable) -> Dict[str, Any]:
    """Extract parameter types from function type hints."""
    return typing.get_type_hints(func)


def extract_tool_info(func: Callable) -> ToolDefinition:
    """Extract tool information from a callable function."""
    # Get function name
    name = func.__name__

    # Get docstring
    description = inspect.getdoc(func) or "No description available"

    # Get parameters and their types
    signature = inspect.signature(func)
    type_hints = extract_type_hints(func)

    parameters = {}
    required_params = []

    for param_name, param in signature.parameters.items():
        # Skip self parameter for methods
        if param_name == "self":
            continue

        param_type = type_hints.get(param_name, Any)

        # Handle optional parameters
        is_optional = (
            param.default != inspect.Parameter.empty
            or getattr(param_type, "__origin__", None) is Union
            and type(None) in param_type.__args__
        )

        if not is_optional:
            required_params.append(param_name)

        parameters[param_name] = {
            "type": str(param_type),
            "default": (
                None
                if param.default is inspect.Parameter.empty
                else param.default
            ),
            "required": not is_optional,
        }

    return ToolDefinition(
        name=name,
        description=description,
        parameters=parameters,
        required_params=required_params,
        callable=func,
    )


class ToolUsingAgent:
    def __init__(
        self,
        tools: List[Callable],
        openai_api_key: str,
        model_name: str = "gpt-4",
        temperature: float = 0.1,
        max_loops: int = 10,
    ):
        # Convert callable tools to ToolDefinitions
        self.available_tools = {
            tool.__name__: extract_tool_info(tool) for tool in tools
        }

        self.execution_plan: List[ExecutionStep] = []
        self.current_step_index = 0
        self.max_loops = max_loops

        # Initialize the OpenAI model
        self.model = OpenAIChat(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=temperature,
        )

        # Create system prompt with tool descriptions
        self.system_prompt = self._create_system_prompt()

        self.agent = Agent(
            agent_name="Tool-Using-Agent",
            system_prompt=self.system_prompt,
            llm=self.model,
            max_loops=1,
            autosave=True,
            verbose=True,
            saved_state_path="tool_agent_state.json",
            context_length=200000,
        )

    def _create_system_prompt(self) -> str:
        """Create system prompt with available tools information."""
        tools_description = []
        for tool_name, tool in self.available_tools.items():
            tools_description.append(
                f"""
                Tool: {tool_name}
                Description: {tool.description}
                Parameters: {json.dumps(tool.parameters, indent=2)}
                Required Parameters: {tool.required_params}
                """
            )

        output = f"""You are an autonomous agent capable of executing complex tasks using available tools.

        Available Tools:
        {chr(10).join(tools_description)}

        Follow these protocols:
        1. Create a detailed plan using available tools
        2. Execute each step in order
        3. Handle errors appropriately
        4. Maintain execution state
        5. Return results in structured format

        You must ALWAYS respond in the following JSON format:
        {{
            "plan": {{
                "description": "Brief description of the overall plan",
                "steps": [
                    {{
                        "step_number": 1,
                        "tool_name": "name_of_tool",
                        "description": "What this step accomplishes",
                        "parameters": {{
                            "param1": "value1",
                            "param2": "value2"
                        }},
                        "expected_output": "Description of expected output"
                    }}
                ]
            }},
            "reasoning": "Explanation of why this plan was chosen"
        }}

        Before executing any tool:
        1. Validate all required parameters are present
        2. Verify parameter types match specifications
        3. Check parameter values are within valid ranges/formats
        4. Ensure logical dependencies between steps are met

        If any validation fails:
        1. Return error in JSON format with specific details
        2. Suggest corrections if possible
        3. Do not proceed with execution

        After each step execution:
        1. Verify output matches expected format
        2. Log results and any warnings/errors
        3. Update execution state
        4. Determine if plan adjustment needed

        Error Handling:
        1. Catch and classify all errors
        2. Provide detailed error messages
        3. Suggest recovery steps
        4. Maintain system stability

        The final output must be valid JSON that can be parsed. Always check your response can be parsed as JSON before returning.
        """
        return output

    def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Any:
        """Execute a tool with given parameters."""
        tool = self.available_tools[tool_name]
        if not tool.callable:
            raise ValueError(
                f"Tool {tool_name} has no associated callable"
            )

        # Convert parameters to appropriate types
        converted_params = {}
        for param_name, param_value in parameters.items():
            param_info = tool.parameters[param_name]
            param_type = eval(
                param_info["type"]
            )  # Note: Be careful with eval
            converted_params[param_name] = param_type(param_value)

        return tool.callable(**converted_params)

    def run(self, task: str) -> Dict[str, Any]:
        """Execute the complete task with proper logging and error handling."""
        execution_log = {
            "task": task,
            "start_time": datetime.utcnow().isoformat(),
            "steps": [],
            "final_result": None
        }
        
        try:
            # Create and execute plan
            plan_response = self.agent.run(f"Create a plan for: {task}")
            plan_data = json.loads(plan_response)
            
            # Extract steps from the correct path in JSON
            steps = plan_data["plan"]["steps"]  # Changed from plan_data["steps"]
            
            for step in steps:
                try:
                    # Check if parameters need default values
                    for param_name, param_value in step["parameters"].items():
                        if isinstance(param_value, str) and not param_value.replace(".", "").isdigit():
                            # If parameter is a description rather than a value, set default
                            if "income" in param_name.lower():
                                step["parameters"][param_name] = 75000.0
                            elif "year" in param_name.lower():
                                step["parameters"][param_name] = 2024
                            elif "investment" in param_name.lower():
                                step["parameters"][param_name] = 1000.0
                    
                    # Execute the tool
                    result = self.execute_tool(
                        step["tool_name"],
                        step["parameters"]
                    )
                    
                    execution_log["steps"].append({
                        "step_number": step["step_number"],
                        "tool": step["tool_name"],
                        "parameters": step["parameters"],
                        "success": True,
                        "result": result,
                        "description": step["description"]
                    })
                    
                except Exception as e:
                    execution_log["steps"].append({
                        "step_number": step["step_number"],
                        "tool": step["tool_name"],
                        "parameters": step["parameters"],
                        "success": False,
                        "error": str(e),
                        "description": step["description"]
                    })
                    print(f"Error executing step {step['step_number']}: {str(e)}")
                    # Continue with next step instead of raising
                    continue
            
            # Only mark as success if at least some steps succeeded
            successful_steps = [s for s in execution_log["steps"] if s["success"]]
            if successful_steps:
                execution_log["final_result"] = {
                    "success": True,
                    "results": successful_steps,
                    "reasoning": plan_data.get("reasoning", "No reasoning provided")
                }
            else:
                execution_log["final_result"] = {
                    "success": False,
                    "error": "No steps completed successfully",
                    "plan": plan_data
                }
                
        except Exception as e:
            execution_log["final_result"] = {
                "success": False,
                "error": str(e),
                "plan": plan_data if 'plan_data' in locals() else None
            }
        
        execution_log["end_time"] = datetime.utcnow().isoformat()
        return execution_log


# Example usage
if __name__ == "__main__":
    load_dotenv()

    # Example tool functions
    def research_ira_requirements() -> Dict[str, Any]:
        """Research and return ROTH IRA eligibility requirements."""
        return {
            "age_requirement": "Must have earned income",
            "income_limits": {"single": 144000, "married": 214000},
        }

    def calculate_contribution_limit(
        income: float, tax_year: int
    ) -> Dict[str, float]:
        """Calculate maximum ROTH IRA contribution based on income and tax year."""
        base_limit = 6000 if tax_year <= 2022 else 6500
        if income > 144000:
            return {"limit": 0}
        return {"limit": base_limit}

    def find_brokers(min_investment: float) -> List[Dict[str, Any]]:
        """Find suitable brokers for ROTH IRA based on minimum investment."""
        return [
            {"name": "Broker A", "min_investment": min_investment},
            {
                "name": "Broker B",
                "min_investment": min_investment * 1.5,
            },
        ]

    # Initialize agent with tools
    agent = ToolUsingAgent(
        tools=[
            research_ira_requirements,
            calculate_contribution_limit,
            find_brokers,
        ],
        openai_api_key="",
    )

    # Run a task
    result = agent.run(
        "How can I establish a ROTH IRA to buy stocks and get a tax break? "
        "What are the criteria?"
    )

    print(json.dumps(result, indent=2))

# The Swarms Tool System: Functions, Pydantic BaseModels as Tools, and Radical Customization


This guide provides an in-depth look at the Swarms Tool System, focusing on its functions, the use of Pydantic BaseModels as tools, and the extensive customization options available. Aimed at developers, this documentation highlights how the Swarms framework works and offers detailed examples of creating and customizing tools and agents, specifically for accounting tasks.

The Swarms Tool System is a flexible and extensible component of the Swarms framework that allows for the creation, registration, and utilization of various tools. These tools can perform a wide range of tasks and are integrated into agents to provide specific functionalities. The system supports multiple ways to define tools, including using Pydantic BaseModels, functions, and dictionaries.

### Architecture

The architecture of the Swarms Tool System is designed to be highly modular. It consists of the following main components:

1. **Agents:** The primary entities that execute tasks.
2. **Tools:** Functions or classes that perform specific operations.
3. **Schemas:** Definitions of input and output data formats using Pydantic BaseModels.

### Key Concepts

#### Tools

Tools are the core functional units within the Swarms framework. They can be defined in various ways:

- **Pydantic BaseModels**: Tools can be defined using Pydantic BaseModels to ensure data validation and serialization.
- **Functions**: Tools can be simple or complex functions.
- **Dictionaries**: Tools can be represented as dictionaries for flexibility.

#### Agents

Agents utilize tools to perform tasks. They are configured with a set of tools and schemas, and they execute the tools based on the input they receive.

## Detailed Documentation

### Tool Definition

#### Using Pydantic BaseModels

Pydantic BaseModels provide a structured way to define tool inputs and outputs. They ensure data validation and serialization, making them ideal for complex data handling.

**Example:**

Define Pydantic BaseModels for accounting tasks:

```python
from pydantic import BaseModel

class CalculateTax(BaseModel):
    income: float

class GenerateInvoice(BaseModel):
    client_name: str
    amount: float
    date: str

class SummarizeExpenses(BaseModel):
    expenses: list[dict]
```

Define tool functions using these models:

```python
def calculate_tax(data: CalculateTax) -> dict:
    tax_rate = 0.3  # Example tax rate
    tax = data.income * tax_rate
    return {"income": data.income, "tax": tax}

def generate_invoice(data: GenerateInvoice) -> dict:
    invoice = {
        "client_name": data.client_name,
        "amount": data.amount,
        "date": data.date,
        "invoice_id": "INV12345"
    }
    return invoice

def summarize_expenses(data: SummarizeExpenses) -> dict:
    total_expenses = sum(expense['amount'] for expense in data.expenses)
    return {"total_expenses": total_expenses}
```

#### Using Functions Directly

Tools can also be defined directly as functions without using Pydantic models. This approach is suitable for simpler tasks where complex validation is not required.

**Example:**

```python
def basic_tax_calculation(income: float) -> dict:
    tax_rate = 0.25
    tax = income * tax_rate
    return {"income": income, "tax": tax}
```

#### Using Dictionaries

Tools can be represented as dictionaries, providing maximum flexibility. This method is useful when the tool's functionality is more dynamic or when integrating with external systems.

**Example:**

```python
basic_tool_schema = {
    "name": "basic_tax_tool",
    "description": "A basic tax calculation tool",
    "parameters": {
        "type": "object",
        "properties": {
            "income": {"type": "number", "description": "Income amount"}
        },
        "required": ["income"]
    }
}

def basic_tax_tool(income: float) -> dict:
    tax_rate = 0.2
    tax = income * tax_rate
    return {"income": income, "tax": tax}
```

### Tool Registration

Tools need to be registered with the agent for it to utilize them. This can be done by specifying the tools in the `tools` parameter during agent initialization.

**Example:**

```python
from swarms import Agent
from llama_hosted import llama3Hosted

# Define Pydantic BaseModels for accounting tasks
class CalculateTax(BaseModel):
    income: float

class GenerateInvoice(BaseModel):
    client_name: str
    amount: float
    date: str

class SummarizeExpenses(BaseModel):
    expenses: list[dict]

# Define tool functions using these models
def calculate_tax(data: CalculateTax) -> dict:
    tax_rate = 0.3
    tax = data.income * tax_rate
    return {"income": data.income, "tax": tax}

def generate_invoice(data: GenerateInvoice) -> dict:
    invoice = {
        "client_name": data.client_name,
        "amount": data.amount,
        "date": data.date,
        "invoice_id": "INV12345"
    }
    return invoice

def summarize_expenses(data: SummarizeExpenses) -> dict:
    total_expenses = sum(expense['amount'] for expense in data.expenses)
    return {"total_expenses": total_expenses}

# Function to generate a tool schema for demonstration purposes
def create_tool_schema():
    return {
        "name": "execute",
        "description": "Executes code on the user's machine",
        "parameters": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "description": "Programming language",
                    "enum": ["python", "java"]
                },
                "code": {"type": "string", "description": "Code to execute"}
            },
            "required": ["language", "code"]
        }
    }

# Initialize the agent with the tools
agent = Agent(
    agent_name="Accounting Agent",
    system_prompt="This agent assists with various accounting tasks.",
    sop_list=["Provide accurate and timely accounting services."],
    llm=llama3Hosted(),
    max_loops="auto",
    interactive=True,
    verbose=True,
    tool_schema=BaseModel,
    list_base_models=[
        CalculateTax,
        GenerateInvoice,
        SummarizeExpenses
    ],
    output_type=str,
    metadata_output_type="json",
    function_calling_format_type="OpenAI",
    function_calling_type="json",
    tools=[
        calculate_tax,
        generate_invoice,
        summarize_expenses
    ],
    list_tool_schemas_json=create_tool_schema(),
)
```

### Running the Agent

The agent can execute tasks using the `run` method. This method takes a prompt and determines the appropriate tool to use based on the input.

**Example:**

```python
# Example task: Calculate tax for an income
result = agent.run("Calculate the tax for an income of $50,000.")
print(f"Result: {result}")

# Example task: Generate an invoice
invoice_data = agent.run("Generate an invoice for John Doe for $1500 on 2024-06-01.")
print(f"Invoice Data: {invoice_data}")

# Example task: Summarize expenses
expenses = [
    {"amount": 200.0, "description": "Office supplies"},
    {"amount": 1500.0, "description": "Software licenses"},
    {"amount": 300.0, "description": "Travel expenses"}
]
summary = agent.run("Summarize these expenses: " + str(expenses))
print(f"Expenses Summary: {summary}")
```


### Customizing Tools

Custom tools can be created to extend the functionality of the Swarms framework. This can include integrating external APIs, performing complex calculations, or handling specialized data formats.

**Example: Custom Accounting Tool**

```python
from pydantic import BaseModel

class CustomAccountingTool(BaseModel):
    data: dict

def custom_accounting_tool(data: CustomAccountingTool) -> dict:
    # Custom logic for the accounting tool
    result = {
        "status": "success",
        "data_processed": len(data.data)
    }
    return result

# Register the custom tool with the agent
agent = Agent(
    agent_name="Accounting Agent",
    system_prompt="This agent assists with various accounting tasks.",
    sop_list=["Provide accurate and timely accounting services."],
    llm=llama3Hosted(),
    max_loops="auto",
    interactive=True,
    verbose=True,
    tool_schema=BaseModel,
    list_base_models=[
        CalculateTax,
        GenerateInvoice,
        SummarizeExpenses,
        CustomAccountingTool
    ],
    output_type=str,
    metadata_output_type="json",
    function_calling_format_type="OpenAI",
    function_calling_type="json",
    tools=[
        calculate_tax,
        generate_invoice,
        summarize_expenses,
        custom_accounting_tool
    ],
    list_tool_schemas_json=create_tool_schema(),
)
```

### Advanced Customization

Advanced customization involves modifying the core components of the Swarms framework. This includes extending existing classes, adding new methods, or integrating third-party libraries.

**Example: Extending the Agent Class**

```python
from swarms import Agent

class AdvancedAccountingAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def custom_behavior(self):
        print("Executing custom behavior")

    def another_custom_method(self):
        print("Another

 custom method")

# Initialize the advanced agent
advanced_agent = AdvancedAccountingAgent(
    agent_name="Advanced Accounting Agent",
    system_prompt="This agent performs advanced accounting tasks.",
    sop_list=["Provide advanced accounting services."],
    llm=llama3Hosted(),
    max_loops="auto",
    interactive=True,
    verbose=True,
    tool_schema=BaseModel,
    list_base_models=[
        CalculateTax,
        GenerateInvoice,
        SummarizeExpenses,
        CustomAccountingTool
    ],
    output_type=str,
    metadata_output_type="json",
    function_calling_format_type="OpenAI",
    function_calling_type="json",
    tools=[
        calculate_tax,
        generate_invoice,
        summarize_expenses,
        custom_accounting_tool
    ],
    list_tool_schemas_json=create_tool_schema(),
)

# Call custom methods
advanced_agent.custom_behavior()
advanced_agent.another_custom_method()
```

### Integrating External Libraries

You can integrate external libraries to extend the functionality of your tools. This is useful for adding new capabilities or leveraging existing libraries for complex tasks.

**Example: Integrating Pandas for Data Processing**

```python
import pandas as pd
from pydantic import BaseModel

class DataFrameTool(BaseModel):
    data: list[dict]

def process_data_frame(data: DataFrameTool) -> dict:
    df = pd.DataFrame(data.data)
    summary = df.describe().to_dict()
    return {"summary": summary}

# Register the tool with the agent
agent = Agent(
    agent_name="Data Processing Agent",
    system_prompt="This agent processes data frames.",
    sop_list=["Provide data processing services."],
    llm=llama3Hosted(),
    max_loops="auto",
    interactive=True,
    verbose=True,
    tool_schema=BaseModel,
    list_base_models=[DataFrameTool],
    output_type=str,
    metadata_output_type="json",
    function_calling_format_type="OpenAI",
    function_calling_type="json",
    tools=[process_data_frame],
    list_tool_schemas_json=create_tool_schema(),
)

# Example task: Process a data frame
data = [
    {"col1": 1, "col2": 2},
    {"col1": 3, "col2": 4},
    {"col1": 5, "col2": 6}
]
result = agent.run("Process this data frame: " + str(data))
print(f"Data Frame Summary: {result}")
```

## Conclusion

The Swarms Tool System provides a robust and flexible framework for defining and utilizing tools within agents. By leveraging Pydantic BaseModels, functions, and dictionaries, developers can create highly customized tools to perform a wide range of tasks. The extensive customization options allow for the integration of external libraries and the extension of core components, making the Swarms framework suitable for diverse applications.

This guide has covered the fundamental concepts and provided detailed examples to help you get started with the Swarms Tool System. With this foundation, you can explore and implement advanced features to build powerful
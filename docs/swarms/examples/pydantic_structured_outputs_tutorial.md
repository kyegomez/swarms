# Pydantic Structured Outputs Tutorial

!!! abstract "Overview"
    This comprehensive tutorial provides enterprise-grade guidance for implementing Pydantic BaseModel structured outputs with Swarms agents. Learn how to define schemas, configure agents, and parse responses in production environments.

## What are Pydantic Structured Outputs?

Pydantic structured outputs enable you to define strongly-typed data schemas that language models must follow when generating responses. Instead of receiving free-form text, you obtain validated, structured data that conforms to your exact specifications.

**Key Benefits:**

- **Type Safety**: Automatic validation ensures data matches your schema
- **Consistency**: Guaranteed output format every time
- **Easy Parsing**: Direct access to structured data without string parsing
- **IDE Support**: Full autocomplete and type hints in your code

---

## Step-by-Step Implementation Guide

### Step 1: Define Your Pydantic Models

First, create Pydantic models that define the structure of your desired output.

```python title="Step 1: Define Pydantic Models"
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# Example 1: Simple User Information Model
class UserInfo(BaseModel):
    """User information structure"""
    name: str = Field(description="Full name of the user")
    email: str = Field(description="Email address")
    age: int = Field(description="Age in years", ge=0, le=150)
    is_active: bool = Field(description="Whether the user account is active")

# Example 2: Financial Transaction Model
class Transaction(BaseModel):
    """Financial transaction details"""
    transaction_id: str = Field(description="Unique transaction identifier")
    amount: float = Field(description="Transaction amount", gt=0)
    currency: str = Field(description="Currency code (e.g., USD, EUR)", default="USD")
    timestamp: datetime = Field(description="Transaction timestamp")
    category: str = Field(description="Transaction category")
    merchant: Optional[str] = Field(description="Merchant name", default=None)

# Example 3: Complex Nested Model
class FinancialReport(BaseModel):
    """Comprehensive financial report"""
    report_id: str = Field(description="Unique report identifier")
    generated_at: datetime = Field(description="Report generation timestamp")
    total_income: float = Field(description="Total income amount", ge=0)
    total_expenses: float = Field(description="Total expenses amount", ge=0)
    transactions: List[Transaction] = Field(description="List of transactions")
    summary: str = Field(description="Text summary of the report")
```

!!! tip "Field Descriptions"
    Always include detailed `Field(description=...)` for each field. The LLM uses these descriptions to understand what data to extract and how to format it.

---

### Step 2: Single Model with `tool_schema`

For a single structured output, use the `tool_schema` parameter.

```python title="Step 2: Single Model Configuration"
from swarms import Agent
from pydantic import BaseModel, Field

# Define your model
class ProductInfo(BaseModel):
    """Product information structure"""
    name: str = Field(description="Product name")
    price: float = Field(description="Product price in USD", gt=0)
    category: str = Field(description="Product category")
    in_stock: bool = Field(description="Whether product is in stock")
    description: str = Field(description="Product description")

# Initialize agent with single model
agent = Agent(
    model_name="gpt-4o",
    agent_name="Product-Info-Agent",
    agent_description="Extracts product information from text",
    system_prompt="You are a helpful assistant that extracts product information from user queries.",
    tool_schema=ProductInfo,  # Single Pydantic model
    max_loops=1,
    verbose=True
)

# Run the agent
response = agent.run(
    "Extract product information: iPhone 15 Pro, $999, Electronics, In Stock, Latest iPhone with A17 chip"
)

# The response will be a ProductInfo instance or dict
print(response)
# Output: {'name': 'iPhone 15 Pro', 'price': 999.0, 'category': 'Electronics', ...}
```

!!! note "Response Format"
    The agent automatically converts the Pydantic model response to a dictionary. You can access it directly or convert it back to a model instance.

---

### Step 3: Multiple Models with `list_base_models`

For multiple possible output structures, use `list_base_models` to provide a list of Pydantic models.

```python title="Step 3: Multiple Models Configuration"
from swarms import Agent
from pydantic import BaseModel, Field
from typing import List

# Define multiple models
class EmailExtraction(BaseModel):
    """Email information extraction"""
    sender: str = Field(description="Email sender name")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")
    priority: str = Field(description="Priority level: low, medium, high")

class CalendarEvent(BaseModel):
    """Calendar event information"""
    title: str = Field(description="Event title")
    start_time: str = Field(description="Event start time")
    end_time: str = Field(description="Event end time")
    location: str = Field(description="Event location")
    attendees: List[str] = Field(description="List of attendee names")

class TaskReminder(BaseModel):
    """Task reminder information"""
    task_name: str = Field(description="Task name")
    due_date: str = Field(description="Due date")
    priority: str = Field(description="Task priority")
    notes: str = Field(description="Additional notes")

# Initialize agent with multiple models
agent = Agent(
    model_name="gpt-4o",
    agent_name="Multi-Structured-Agent",
    agent_description="Extracts structured information from various input types",
    system_prompt="""You are a helpful assistant that extracts structured information.
    Based on the user's query, determine which type of information they want:
    - EmailExtraction: For email-related queries
    - CalendarEvent: For calendar/scheduling queries
    - TaskReminder: For task/reminder queries
    """,
    list_base_models=[EmailExtraction, CalendarEvent, TaskReminder],  # Multiple models
    max_loops=1,
    verbose=True
)

# Run the agent - LLM will choose the appropriate model
response = agent.run(
    "Extract calendar event: Team Meeting, Tomorrow 2pm-3pm, Conference Room A, with John and Sarah"
)

print(response)
# Output: {'title': 'Team Meeting', 'start_time': 'Tomorrow 2pm', ...}
```

!!! success "Model Selection"
    When using `list_base_models`, the LLM automatically selects the most appropriate model based on the query context.

---

### Step 4: Complete Example - Financial Analysis Agent

Here's a complete, production-ready example combining all concepts:

```python title="Step 4: Complete Financial Agent Example"
import os
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from swarms import Agent

# Step 4.1: Define comprehensive Pydantic models
class StockAnalysis(BaseModel):
    """Stock market analysis structure"""
    ticker: str = Field(description="Stock ticker symbol")
    current_price: float = Field(description="Current stock price", gt=0)
    price_change: float = Field(description="Price change from previous close")
    price_change_percent: float = Field(description="Percentage change")
    volume: int = Field(description="Trading volume", ge=0)
    market_cap: Optional[float] = Field(description="Market capitalization", default=None)
    recommendation: str = Field(description="Buy, Hold, or Sell recommendation")
    reasoning: str = Field(description="Reasoning for the recommendation")

class PortfolioSummary(BaseModel):
    """Portfolio summary structure"""
    total_value: float = Field(description="Total portfolio value", ge=0)
    total_gain_loss: float = Field(description="Total gain/loss amount")
    total_gain_loss_percent: float = Field(description="Total gain/loss percentage")
    holdings: List[dict] = Field(description="List of holdings with ticker and value")
    risk_level: str = Field(description="Risk level: Low, Medium, High")

class ExpenseCategory(BaseModel):
    """Expense category breakdown"""
    category: str = Field(description="Expense category name")
    amount: float = Field(description="Total amount in this category", ge=0)
    percentage: float = Field(description="Percentage of total expenses", ge=0, le=100)
    transactions_count: int = Field(description="Number of transactions", ge=0)

# Step 4.2: Initialize the agent
agent = Agent(
    model_name="gpt-4o",
    agent_name="Financial-Analyst-Agent",
    agent_description="Professional financial analysis and reporting agent",
    system_prompt="""You are a professional financial analyst. 
    Extract and structure financial information from user queries.
    Provide accurate, well-reasoned financial analysis.
    """,
    list_base_models=[StockAnalysis, PortfolioSummary, ExpenseCategory],
    max_loops=1,
    verbose=True,
    temperature=0.3,  # Lower temperature for more consistent structured outputs
)

# Step 4.3: Run different types of queries
# Query 1: Stock Analysis
stock_query = """
Analyze AAPL stock:
- Current price: $175.50
- Previous close: $173.20
- Change: +$2.30 (+1.33%)
- Volume: 45,234,567
- Market cap: $2.8T
- Based on strong earnings, recommend: Buy
"""

stock_result = agent.run(stock_query)
print("Stock Analysis Result:")
print(stock_result)

# Query 2: Portfolio Summary
portfolio_query = """
Portfolio Summary:
- Total Value: $125,000
- Initial Investment: $100,000
- Gain/Loss: +$25,000 (+25%)
- Holdings: AAPL ($50k), MSFT ($40k), GOOGL ($35k)
- Risk Level: Medium
"""

portfolio_result = agent.run(portfolio_query)
print("\nPortfolio Summary Result:")
print(portfolio_result)

# Query 3: Expense Categories
expense_query = """
Monthly Expenses Breakdown:
- Food & Dining: $800 (32%)
- Transportation: $500 (20%)
- Utilities: $300 (12%)
- Entertainment: $400 (16%)
- Other: $500 (20%)
Total: $2,500
"""

expense_result = agent.run(expense_query)
print("\nExpense Category Result:")
print(expense_result)
```

---

### Step 5: Parsing and Using Structured Outputs

After receiving structured outputs, you can parse and use them in your application:

```python title="Step 5: Parsing Structured Outputs"
from pydantic import BaseModel, ValidationError

# Option 1: Direct dictionary access
response = agent.run("Your query here")
if isinstance(response, dict):
    ticker = response.get("ticker")
    price = response.get("current_price")
    print(f"{ticker}: ${price}")

# Option 2: Convert back to Pydantic model for validation
try:
    stock_analysis = StockAnalysis(**response)
    print(f"Validated: {stock_analysis.ticker} at ${stock_analysis.current_price}")
except ValidationError as e:
    print(f"Validation error: {e}")

# Option 3: Handle list responses (when using multiple models)
if isinstance(response, list):
    for item in response:
        if isinstance(item, dict):
            # Process each structured output
            print(f"Found: {item}")
```

---

### Step 6: Advanced Patterns

#### Pattern 1: Nested Models

```python title="Nested Model Example"
from pydantic import BaseModel, Field
from typing import List

class Address(BaseModel):
    """Address information"""
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    state: str = Field(description="State code")
    zip_code: str = Field(description="ZIP code")

class Person(BaseModel):
    """Person with nested address"""
    name: str = Field(description="Full name")
    age: int = Field(description="Age")
    address: Address = Field(description="Home address")
    contacts: List[str] = Field(description="List of contact methods")

agent = Agent(
    model_name="gpt-4o",
    tool_schema=Person,
    max_loops=1
)
```

#### Pattern 2: Optional Fields and Defaults

```python title="Optional Fields Example"
from pydantic import BaseModel, Field
from typing import Optional

class FlexibleData(BaseModel):
    """Model with optional and required fields"""
    required_field: str = Field(description="This field is always required")
    optional_field: Optional[str] = Field(description="This field may be missing", default=None)
    field_with_default: str = Field(description="Field with default value", default="default_value")
    numeric_optional: Optional[float] = Field(description="Optional number", default=None)
```

#### Pattern 3: Enums and Constraints

```python title="Enums and Constraints Example"
from pydantic import BaseModel, Field
from enum import Enum
from typing import Literal

class Status(str, Enum):
    """Status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class Task(BaseModel):
    """Task with enum and constraints"""
    task_id: str = Field(description="Task identifier")
    status: Status = Field(description="Current task status")
    priority: Literal["low", "medium", "high"] = Field(description="Task priority")
    progress: int = Field(description="Progress percentage", ge=0, le=100)
    estimated_hours: float = Field(description="Estimated hours", gt=0)
```

---

## Best Practices

### 1. Field Descriptions

!!! success "Always Include Descriptions"

    ```python
    # Recommended: Clear, detailed description
    price: float = Field(
        description="Product price in USD. Must be greater than zero."
    )

    # Not Recommended: Missing or vague description
    price: float
    ```

### 2. Validation Constraints

!!! tip "Use Field Constraints"
    ```python
    # Recommended: Use constraints for validation
    age: int = Field(description="Age in years", ge=0, le=150)
    price: float = Field(description="Price", gt=0)
    percentage: float = Field(description="Percentage", ge=0, le=100)
    ```

### 3. Model Organization

!!! note "Organize Related Models"

    ```python
    # Recommended: Group related models together
    # models/financial.py
    class Transaction(BaseModel): ...
    class Portfolio(BaseModel): ...
    class Report(BaseModel): ...

    # models/user.py
    class User(BaseModel): ...
    class Profile(BaseModel): ...
    ```

### 4. Error Handling

!!! warning "Always Validate Responses"

    ```python
    from pydantic import ValidationError

    try:
        result = agent.run(query)
        validated = YourModel(**result)
    except ValidationError as e:
        print(f"Validation failed: {e}")
        # Handle error appropriately
    ```

---

## Common Issues and Solutions

### Issue 1: LLM Returns Wrong Format

!!! failure "Problem"
    The LLM doesn't follow the Pydantic schema and returns unstructured text.

!!! success "Solution"
    - Ensure field descriptions are clear and detailed
    - Use a lower temperature (0.1-0.3) for more consistent outputs
    - Add explicit instructions in the system prompt about following the schema
    - Use models that support structured outputs (gpt-4o, claude-3.5-sonnet, etc.)

### Issue 2: Missing Required Fields

!!! failure "Problem"
    The response is missing required fields from your Pydantic model.

!!! success "Solution"

    ```python
    # Make fields optional if they might be missing
    optional_field: Optional[str] = Field(default=None)

    # Or provide defaults
    field_with_default: str = Field(default="default_value")

    # Or mark as required in system prompt
    system_prompt = "Always include all required fields: field1, field2, field3"
    ```

### Issue 3: Type Mismatches

!!! failure "Problem"
    The LLM returns the wrong data type (e.g., string instead of number).

!!! success "Solution"

    ```python
    # Use explicit type hints and Field descriptions
    price: float = Field(
        description="Price as a decimal number (e.g., 99.99), not a string"
    )

    # Add validation in the model
    from pydantic import validator

    @validator('price')
    def validate_price(cls, v):
        if isinstance(v, str):
            return float(v)
        return v
    ```

---

## Production Use Cases

### Example 1: E-commerce Product Extractor

```python
from pydantic import BaseModel, Field
from swarms import Agent

class Product(BaseModel):
    """E-commerce product information"""
    name: str = Field(description="Product name")
    brand: str = Field(description="Brand name")
    price: float = Field(description="Price in USD", gt=0)
    category: str = Field(description="Product category")
    rating: float = Field(description="Average rating 0-5", ge=0, le=5)
    in_stock: bool = Field(description="Stock availability")
    features: list[str] = Field(description="List of product features")

agent = Agent(
    model_name="gpt-4o",
    tool_schema=Product,
    system_prompt="Extract product information from user queries.",
    max_loops=1
)

result = agent.run(
    "iPhone 15 Pro Max by Apple, $1199, Smartphones, 4.5 stars, "
    "In Stock, Features: A17 Pro chip, 48MP camera, Titanium design"
)
```

### Example 2: Meeting Scheduler

```python
from pydantic import BaseModel, Field
from typing import List
from swarms import Agent

class MeetingRequest(BaseModel):
    """Meeting scheduling request"""
    title: str = Field(description="Meeting title")
    date: str = Field(description="Meeting date (YYYY-MM-DD)")
    start_time: str = Field(description="Start time (HH:MM)")
    end_time: str = Field(description="End time (HH:MM)")
    location: str = Field(description="Meeting location")
    attendees: List[str] = Field(description="List of attendee names")
    agenda: str = Field(description="Meeting agenda")

agent = Agent(
    model_name="gpt-4o",
    tool_schema=MeetingRequest,
    system_prompt="Extract meeting details from scheduling requests.",
    max_loops=1
)

result = agent.run(
    "Schedule a team standup for 2024-01-15 at 10:00 AM to 10:30 AM "
    "in Conference Room B with Alice, Bob, and Charlie. "
    "Agenda: Review sprint progress and blockers."
)
```

---

## Summary

This tutorial covered the following topics:

1. **Pydantic Model Definition**: How to define models with Field descriptions and constraints
2. **Single Model Configuration**: Using the `tool_schema` parameter for single structured outputs
3. **Multiple Models Configuration**: Using the `list_base_models` parameter for multiple output types
4. **Production Examples**: Complete implementations for enterprise use cases
5. **Best Practices**: Guidelines for field descriptions, validation, and error handling
6. **Troubleshooting**: Solutions for common issues with structured outputs

---

## Next Steps

- Explore [Agent Reference Documentation](../structs/agent.md) for advanced configuration options
- Learn about [Tools and MCP Integration](../tools/tools_examples.md)
- Review [Additional Examples](../../examples/index.md) for more use cases

!!! question "Support and Resources"
    For additional assistance:
    - Review the [Troubleshooting Section](#common-issues-and-solutions) above
    - Consult the [Support Documentation](../support.md)
    - Access the [Discord Community](https://discord.gg/EamjgSaEQf) for community support

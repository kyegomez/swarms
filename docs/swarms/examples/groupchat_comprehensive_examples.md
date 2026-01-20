# GroupChat Comprehensive Examples

!!! abstract "Overview"
    This comprehensive guide showcases all features of the GroupChat system, including speaker functions, @mentions, interactive mode, image support, and advanced collaboration patterns.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Basic Setup](#basic-setup)
3. [Speaker Functions](#speaker-functions)
4. [@Mention System](#mention-system)
5. [Interactive Mode](#interactive-mode)
6. [Image Support](#image-support)
7. [Custom Speaker Functions](#custom-speaker-functions)
8. [Output Types](#output-types)
9. [Advanced Features](#advanced-features)
10. [Best Practices](#best-practices)

## Prerequisites

!!! info "Before You Begin"
    Make sure you have:
    - Python 3.10+ installed
    - A valid API key for your model provider
    - The Swarms package installed

```bash
pip install swarms
```

## Basic Setup

### Environment Configuration

```python
# .env file
OPENAI_API_KEY="your-api-key-here"
```

### Import Required Modules

```python
from dotenv import load_dotenv
import os
from swarms import Agent, GroupChat
from swarms.structs.groupchat import (
    round_robin_speaker,
    random_speaker,
    priority_speaker,
    random_dynamic_speaker,
)
```

### Creating Example Agents

```python
# Create a team of specialized agents
analyst = Agent(
    agent_name="analyst",
    system_prompt="You are a data analyst. You excel at analyzing data, creating charts, and providing insights.",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=True,
)

researcher = Agent(
    agent_name="researcher",
    system_prompt="You are a research specialist. You are great at gathering information, fact-checking, and providing detailed research.",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=True,
)

writer = Agent(
    agent_name="writer",
    system_prompt="You are a content writer. You excel at writing clear, engaging content and summarizing information.",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=True,
)

reviewer = Agent(
    agent_name="reviewer",
    system_prompt="You are a quality reviewer. You ensure accuracy, completeness, and quality of all outputs.",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=True,
)

agents = [analyst, researcher, writer, reviewer]
```

## Speaker Functions

### 1. Round Robin Speaker

!!! example "Round Robin - Sequential Order"
    Agents speak in a fixed order, cycling through the list:

```python
group_chat = GroupChat(
    name="Round Robin Team",
    description="A team that speaks in sequential order",
    agents=agents,
    speaker_function="round-robin-speaker",
    max_loops=1,
)

task = "Let's create a comprehensive market analysis report."
response = group_chat.run(task)
print(response)
```

**Example Output:**

```json
{
    "messages": [
        {
            "role": "System",
            "content": "Group Chat Name: Round Robin Team\nDescription: A team that speaks in sequential order\n..."
        },
        {
            "role": "User",
            "content": "Let's create a comprehensive market analysis report."
        },
        {
            "role": "analyst",
            "content": "I'll analyze the market data and provide key insights. Based on current trends..."
        },
        {
            "role": "researcher",
            "content": "Building on @analyst's data insights, I've researched additional market factors..."
        },
        {
            "role": "writer",
            "content": "I've reviewed @analyst's analysis and @researcher's findings. Here's a comprehensive summary..."
        },
        {
            "role": "reviewer",
            "content": "I've reviewed all contributions. The report is comprehensive and accurate..."
        }
    ]
}
```

### 2. Random Speaker

!!! example "Random Speaker - Unpredictable Order"
    Agents are selected randomly to speak:

```python
group_chat = GroupChat(
    name="Random Team",
    description="A team that speaks in random order",
    agents=agents,
    speaker_function="random-speaker",
    max_loops=1,
)

task = "Let's brainstorm ideas for a new product launch."
response = group_chat.run(task)
print(response)
```

**Example Output:**

```json
{
    "messages": [
        {
            "role": "User",
            "content": "Let's brainstorm ideas for a new product launch."
        },
        {
            "role": "writer",
            "content": "For a product launch, I suggest focusing on storytelling and emotional connection..."
        }
    ]
}
```

### 3. Priority Speaker

!!! example "Priority Speaker - Weighted Selection"
    Agents are selected based on priority weights:

```python
# Define priorities (higher number = higher priority)
priorities = {
    "researcher": 0.5,  # 50% chance
    "analyst": 0.3,     # 30% chance
    "writer": 0.15,     # 15% chance
    "reviewer": 0.05,   # 5% chance
}

group_chat = GroupChat(
    name="Priority Team",
    description="A team with weighted speaker selection",
    agents=agents,
    speaker_function="priority-speaker",
    speaker_state={"priorities": priorities},
    max_loops=1,
)

task = "Let's analyze the quarterly financial data."
response = group_chat.run(task)
print(response)
```

**Example Output:**

```json
{
    "messages": [
        {
            "role": "User",
            "content": "Let's analyze the quarterly financial data."
        },
        {
            "role": "researcher",
            "content": "I'll research the financial context and market conditions for Q1..."
        }
    ]
}
```

### 4. Random Dynamic Speaker - Sequential Strategy

!!! example "Random Dynamic Speaker - @Mention Based"
    First agent is random, then follows @mentions in responses:

```python
group_chat = GroupChat(
    name="Dynamic Team",
    description="A team that follows @mentions in responses",
    agents=agents,
    speaker_function="random-dynamic-speaker",
    speaker_state={"strategy": "sequential"},
    max_loops=5,  # Allow multiple turns for dynamic flow
)

task = "Let's create a marketing strategy. @analyst please start by analyzing the market data."
response = group_chat.run(task)
print(response)
```

**Example Output:**

```json
{
    "messages": [
        {
            "role": "User",
            "content": "Let's create a marketing strategy. @analyst please start by analyzing the market data."
        },
        {
            "role": "analyst",
            "content": "I've analyzed the market data. Key findings include strong growth in segment A. @researcher, can you verify these trends?"
        },
        {
            "role": "researcher",
            "content": "I've verified @analyst's findings. The trends are consistent across multiple sources. @writer, please draft the strategy document."
        },
        {
            "role": "writer",
            "content": "Based on @analyst's data and @researcher's verification, I've drafted the marketing strategy..."
        }
    ]
}
```

### 5. Random Dynamic Speaker - Parallel Strategy

!!! example "Parallel Execution"
    Multiple mentioned agents respond simultaneously:

```python
group_chat = GroupChat(
    name="Parallel Team",
    description="A team that can respond in parallel",
    agents=agents,
    speaker_function="random-dynamic-speaker",
    speaker_state={"strategy": "parallel"},
    max_loops=5,
)

task = "Let's analyze this from multiple angles. @analyst @researcher please work together."
response = group_chat.run(task)
print(response)
```

**Example Output:**

```json
{
    "messages": [
        {
            "role": "User",
            "content": "Let's analyze this from multiple angles. @analyst @researcher please work together."
        },
        {
            "role": "analyst",
            "content": "From a data perspective, I see three key metrics..."
        },
        {
            "role": "researcher",
            "content": "From a research perspective, I've found supporting evidence..."
        }
    ]
}
```

## @Mention System

### Basic @Mentions

!!! example "Directing Tasks to Specific Agents"
    Use @agent_name to direct tasks to specific agents:

```python
group_chat = GroupChat(
    name="Mention Team",
    description="A team that responds to @mentions",
    agents=agents,
    speaker_function="round-robin-speaker",
)

# Mention specific agents
task = "@analyst Please analyze the sales data. @writer Please draft a summary."
response = group_chat.run(task)
print(response)
```

**Example Output:**

```json
{
    "messages": [
        {
            "role": "User",
            "content": "@analyst Please analyze the sales data. @writer Please draft a summary."
        },
        {
            "role": "analyst",
            "content": "I've analyzed the sales data. Key findings: revenue increased 15%, top product is X..."
        },
        {
            "role": "writer",
            "content": "Building on @analyst's analysis, here's a summary: Sales performance shows strong growth..."
        }
    ]
}
```

### Multiple @Mentions

!!! example "Mentioning Multiple Agents"
    Mention multiple agents in a single task:

```python
group_chat = GroupChat(
    name="Collaborative Team",
    description="A team for collaborative tasks",
    agents=agents,
    speaker_function="round-robin-speaker",
)

task = "@analyst @researcher @writer Let's work together on a comprehensive market report."
response = group_chat.run(task)
print(response)
```

**Example Output:**

```json
{
    "messages": [
        {
            "role": "User",
            "content": "@analyst @researcher @writer Let's work together on a comprehensive market report."
        },
        {
            "role": "analyst",
            "content": "I'll start by analyzing the quantitative data..."
        },
        {
            "role": "researcher",
            "content": "I've reviewed @analyst's data. Let me add market research context..."
        },
        {
            "role": "writer",
            "content": "Combining @analyst's data and @researcher's insights, here's the comprehensive report..."
        }
    ]
}
```

### Agent-to-Agent Mentions

!!! example "Agents Mentioning Each Other"
    Agents can mention each other in their responses:

```python
group_chat = GroupChat(
    name="Self-Organizing Team",
    description="A team where agents can delegate to each other",
    agents=agents,
    speaker_function="random-dynamic-speaker",
    speaker_state={"strategy": "sequential"},
    max_loops=10,  # Allow for multiple back-and-forth exchanges
)

task = "@analyst Please start by analyzing the data. Feel free to ask @researcher for additional information if needed."
response = group_chat.run(task)
print(response)
```

**Example Output:**

```json
{
    "messages": [
        {
            "role": "User",
            "content": "@analyst Please start by analyzing the data. Feel free to ask @researcher for additional information if needed."
        },
        {
            "role": "analyst",
            "content": "I'm analyzing the data. @researcher, can you provide context on market trends for Q1?"
        },
        {
            "role": "researcher",
            "content": "I've researched Q1 market trends. Key factors include economic indicators and competitor activity. @analyst, this should help with your analysis."
        },
        {
            "role": "analyst",
            "content": "Thanks @researcher! With that context, my analysis shows..."
        }
    ]
}
```

## Interactive Mode

### Starting an Interactive Session

!!! example "Interactive Terminal Session"
    Enable interactive mode for real-time chat:

```python
group_chat = GroupChat(
    name="Interactive Team",
    description="A team for interactive collaboration",
    agents=agents,
    speaker_function="round-robin-speaker",
    interactive=True,  # Enable interactive mode
)

# Start the interactive session
group_chat.start_interactive_session()
```

**Example Interactive Session Output:**

```text
Welcome to Interactive Team!
Description: A team for interactive collaboration
Current speaker function: round-robin-speaker

Available agents:
- @analyst: You are a data analyst. You excel at analyzing data...
- @researcher: You are a research specialist. You are great at gathering information...
- @writer: You are a content writer. You excel at writing clear, engaging content...
- @reviewer: You are a quality reviewer. You ensure accuracy, completeness...

Commands:
- Type 'help' or '?' for help
- Type 'exit' or 'quit' to end the session
- Type 'speaker' to change speaker function
- Use @agent_name to mention specific agents (optional)

Start chatting:

You: @analyst analyze the sales data
analyst: I've analyzed the sales data. Key metrics show...

You: @writer create a summary
writer: Based on @analyst's analysis, here's a summary...

You: exit
Goodbye!
```

### Interactive Session Commands

!!! tip "Available Commands"
    - `help` or `?` - Show help information
    - `speaker` - Change the speaker function
    - `exit` or `quit` - End the session
    - `@agent_name` - Mention specific agents (optional)

## Image Support

### Single Image Input

!!! example "GroupChat with Image"
    Pass a single image to the group chat:

```python
group_chat = GroupChat(
    name="Vision Team",
    description="A team that can analyze images",
    agents=agents,
    speaker_function="round-robin-speaker",
)

# Path to image file
image_path = "path/to/image.png"

task = "@analyst Please analyze this chart. @writer Please summarize the findings."
response = group_chat.run(task, img=image_path)
print(response)
```

**Example Output:**

```json
{
    "messages": [
        {
            "role": "User",
            "content": "@analyst Please analyze this chart. @writer Please summarize the findings."
        },
        {
            "role": "analyst",
            "content": "I've analyzed the chart. It shows sales trends over the past 12 months with a peak in Q3..."
        },
        {
            "role": "writer",
            "content": "Based on @analyst's analysis of the chart, here's a summary: The data visualization reveals..."
        }
    ]
}
```

### Multiple Images Input

!!! example "GroupChat with Multiple Images"
    Pass multiple images to the group chat:

```python
group_chat = GroupChat(
    name="Multi-Vision Team",
    description="A team that can analyze multiple images",
    agents=agents,
    speaker_function="round-robin-speaker",
)

# List of image paths
image_paths = [
    "path/to/image1.png",
    "path/to/image2.png",
    "path/to/image3.png",
]

task = "@analyst Please compare these charts. @researcher Please provide context."
response = group_chat.run(task, imgs=image_paths)
print(response)
```

**Example Output:**

```json
{
    "messages": [
        {
            "role": "User",
            "content": "@analyst Please compare these charts. @researcher Please provide context."
        },
        {
            "role": "analyst",
            "content": "Comparing the three charts: Chart 1 shows Q1 data, Chart 2 shows Q2, and Chart 3 shows Q3..."
        },
        {
            "role": "researcher",
            "content": "Based on @analyst's comparison, the context suggests these charts represent quarterly performance trends..."
        }
    ]
}
```

## Custom Speaker Functions

### Creating a Custom Speaker Function

!!! example "Custom Speaker Function"
    Create your own speaker selection logic:

```python
from typing import List, Union

def custom_expertise_speaker(agents: List[str], **kwargs) -> str:
    """Custom speaker function that selects agents based on task keywords."""
    task = kwargs.get("task", "").lower()
    
    # Map keywords to agents
    if "data" in task or "analyze" in task:
        if "analyst" in agents:
            return "analyst"
    elif "research" in task or "information" in task:
        if "researcher" in agents:
            return "researcher"
    elif "write" in task or "content" in task:
        if "writer" in agents:
            return "writer"
    
    # Default to first agent
    return agents[0] if agents else None

group_chat = GroupChat(
    name="Custom Team",
    description="A team with custom speaker selection",
    agents=agents,
    speaker_function=custom_expertise_speaker,
)

task = "Let's analyze the quarterly sales data."
response = group_chat.run(task)
print(response)
```

**Example Output:**

```json
{
    "messages": [
        {
            "role": "User",
            "content": "Let's analyze the quarterly sales data."
        },
        {
            "role": "analyst",
            "content": "I've analyzed the quarterly sales data. Key findings include..."
        }
    ]
}
```

### Advanced Custom Speaker Function

!!! example "Complex Custom Logic"
    Create a more sophisticated custom speaker function:

```python
import re
from typing import List, Union

def advanced_custom_speaker(
    agents: List[str],
    response: str = "",
    history: List[str] = None,
    **kwargs
) -> Union[str, List[str]]:
    """Advanced custom speaker function with history awareness."""
    if not response:
        # First turn: select based on task
        task = kwargs.get("task", "").lower()
        if "analyze" in task:
            return "analyst" if "analyst" in agents else agents[0]
        return agents[0]
    
    # Extract mentions from response
    mentions = re.findall(r"@(\w+)", response)
    valid_mentions = [m for m in mentions if m in agents]
    
    if valid_mentions:
        return valid_mentions[0] if len(valid_mentions) == 1 else valid_mentions
    
    # If no mentions, rotate through agents
    if history:
        last_speaker = history[-1].split(":")[0] if ":" in history[-1] else None
        if last_speaker in agents:
            current_idx = agents.index(last_speaker)
            next_idx = (current_idx + 1) % len(agents)
            return agents[next_idx]
    
    return agents[0]

group_chat = GroupChat(
    name="Advanced Custom Team",
    description="A team with advanced custom speaker selection",
    agents=agents,
    speaker_function=advanced_custom_speaker,
    max_loops=5,
)

task = "Let's create a comprehensive analysis."
response = group_chat.run(task)
print(response)
```

**Example Output:**

```json
{
    "messages": [
        {
            "role": "User",
            "content": "Let's create a comprehensive analysis."
        },
        {
            "role": "analyst",
            "content": "I'll start the analysis. @researcher, can you provide market context?"
        },
        {
            "role": "researcher",
            "content": "I've gathered market context. @writer, please compile the findings."
        },
        {
            "role": "writer",
            "content": "I've compiled the comprehensive analysis based on @analyst's data and @researcher's context."
        }
    ]
}
```

## Output Types

### Dictionary Output

!!! example "Dictionary Format"
    Get conversation history as a dictionary:

```python
group_chat = GroupChat(
    name="Dict Output Team",
    description="A team with dictionary output",
    agents=agents,
    output_type="dict",
)

task = "Let's discuss the project plan."
response = group_chat.run(task)
print(type(response))  # <class 'dict'>
print(response)
```

**Example Output:**

```text
<class 'dict'>
{
    "messages": [
        {
            "role": "System",
            "content": "Group Chat Name: Dict Output Team..."
        },
        {
            "role": "User",
            "content": "Let's discuss the project plan."
        },
        {
            "role": "analyst",
            "content": "For the project plan, I recommend starting with data analysis..."
        }
    ]
}
```

### String Output

!!! example "String Format"
    Get conversation history as a formatted string:

```python
group_chat = GroupChat(
    name="String Output Team",
    description="A team with string output",
    agents=agents,
    output_type="str",
)

task = "Let's discuss the project plan."
response = group_chat.run(task)
print(type(response))  # <class 'str'>
print(response)
```

**Example Output:**

```text
<class 'str'>
System: Group Chat Name: String Output Team
Description: A team with string output

User: Let's discuss the project plan.

analyst: For the project plan, I recommend starting with data analysis to understand the baseline metrics...
```

### All Output Types

!!! example "All Output Format"
    Get complete conversation history:

```python
group_chat = GroupChat(
    name="All Output Team",
    description="A team with complete output",
    agents=agents,
    output_type="all",
)

task = "Let's discuss the project plan."
response = group_chat.run(task)
print(response)
```

**Example Output:**

```json
{
    "messages": [...],
    "metadata": {
        "name": "All Output Team",
        "description": "A team with complete output",
        "total_messages": 3
    }
}
```

## Advanced Features

### Changing Speaker Function at Runtime

!!! example "Dynamic Speaker Function Change"
    Change the speaker function during execution:

```python
group_chat = GroupChat(
    name="Dynamic Team",
    description="A team with changeable speaker function",
    agents=agents,
    speaker_function="round-robin-speaker",
)

# Run with round robin
task1 = "Let's start with round robin order."
response1 = group_chat.run(task1)
print("Round Robin Output:")
print(response1)

# Change to random
group_chat.set_speaker_function("random-speaker")
task2 = "Now let's use random order."
response2 = group_chat.run(task2)
print("\nRandom Output:")
print(response2)

# Change to priority
priorities = {"analyst": 0.5, "researcher": 0.3, "writer": 0.15, "reviewer": 0.05}
group_chat.set_speaker_function("priority-speaker", speaker_state={"priorities": priorities})
task3 = "Now with priority weights."
response3 = group_chat.run(task3)
print("\nPriority Output:")
print(response3)
```

**Example Output:**

```text
Round Robin Output:
{
    "messages": [
        {"role": "User", "content": "Let's start with round robin order."},
        {"role": "analyst", "content": "Starting analysis..."},
        {"role": "researcher", "content": "Building on @analyst's work..."}
    ]
}

Random Output:
{
    "messages": [
        {"role": "User", "content": "Now let's use random order."},
        {"role": "writer", "content": "I'll draft the content..."}
    ]
}

Priority Output:
{
    "messages": [
        {"role": "User", "content": "Now with priority weights."},
        {"role": "analyst", "content": "Analyzing with priority weighting..."}
    ]
}
```

### Conversation History Management

!!! example "Accessing Conversation History"
    Access and manipulate conversation history:

```python
group_chat = GroupChat(
    name="History Team",
    description="A team with conversation history",
    agents=agents,
)

# Run multiple tasks
task1 = "Let's discuss the first topic."
response1 = group_chat.run(task1)
print("First Response:")
print(response1)

task2 = "Now let's discuss the second topic."
response2 = group_chat.run(task2)
print("\nSecond Response:")
print(response2)

# Access full conversation history
history = group_chat.conversation.return_history_as_string()
print("\nFull History:")
print(history)

# Access as list
history_list = group_chat.conversation.messages
print("\nHistory List:")
print(history_list)
```

**Example Output:**

```text
First Response:
{
    "messages": [
        {"role": "User", "content": "Let's discuss the first topic."},
        {"role": "analyst", "content": "First topic analysis..."}
    ]
}

Second Response:
{
    "messages": [
        {"role": "User", "content": "Let's discuss the first topic."},
        {"role": "analyst", "content": "First topic analysis..."},
        {"role": "User", "content": "Now let's discuss the second topic."},
        {"role": "researcher", "content": "Second topic research..."}
    ]
}

Full History:
System: Group Chat Name: History Team...
User: Let's discuss the first topic.
analyst: First topic analysis...
User: Now let's discuss the second topic.
researcher: Second topic research...

History List:
[
    {"role": "System", "content": "..."},
    {"role": "User", "content": "Let's discuss the first topic."},
    {"role": "analyst", "content": "First topic analysis..."},
    {"role": "User", "content": "Now let's discuss the second topic."},
    {"role": "researcher", "content": "Second topic research..."}
]
```

### Batched Execution

!!! example "Running Multiple Tasks"
    Process multiple tasks in sequence:

```python
group_chat = GroupChat(
    name="Batch Team",
    description="A team for batch processing",
    agents=agents,
)

tasks = [
    "Analyze the Q1 sales data.",
    "Research market trends for Q2.",
    "Write a summary report.",
]

responses = group_chat.batched_run(tasks)
for i, response in enumerate(responses):
    print(f"Task {i+1} Response:")
    print(response)
    print("\n" + "="*50 + "\n")
```

**Example Output:**

```text
Task 1 Response:
{
    "messages": [
        {"role": "User", "content": "Analyze the Q1 sales data."},
        {"role": "analyst", "content": "Q1 sales analysis shows..."}
    ]
}

==================================================

Task 2 Response:
{
    "messages": [
        {"role": "User", "content": "Research market trends for Q2."},
        {"role": "researcher", "content": "Q2 market trends research indicates..."}
    ]
}

==================================================

Task 3 Response:
{
    "messages": [
        {"role": "User", "content": "Write a summary report."},
        {"role": "writer", "content": "Summary report: Based on Q1 analysis and Q2 trends..."}
    ]
}
```

### Concurrent Execution

!!! example "Parallel Task Processing"
    Process multiple tasks concurrently:

```python
group_chat = GroupChat(
    name="Concurrent Team",
    description="A team for concurrent processing",
    agents=agents,
)

tasks = [
    "Analyze the Q1 sales data.",
    "Research market trends for Q2.",
    "Write a summary report.",
]

responses = group_chat.concurrent_run(tasks)
for i, response in enumerate(responses):
    print(f"Task {i+1} Response:")
    print(response)
    print("\n" + "="*50 + "\n")
```

**Example Output:**

```text
Task 1 Response:
{
    "messages": [
        {"role": "User", "content": "Analyze the Q1 sales data."},
        {"role": "analyst", "content": "Q1 sales analysis shows..."}
    ]
}

==================================================

Task 2 Response:
{
    "messages": [
        {"role": "User", "content": "Research market trends for Q2."},
        {"role": "researcher", "content": "Q2 market trends research indicates..."}
    ]
}

==================================================

Task 3 Response:
{
    "messages": [
        {"role": "User", "content": "Write a summary report."},
        {"role": "writer", "content": "Summary report: Based on Q1 analysis and Q2 trends..."}
    ]
}
```

### Rules and Guidelines

!!! example "Adding Conversation Rules"
    Define rules for the conversation:

```python
rules = """
1. Always acknowledge other agents' contributions
2. Build upon previous responses
3. Be concise and clear
4. Use @mentions when delegating tasks
"""

group_chat = GroupChat(
    name="Rules Team",
    description="A team with conversation rules",
    agents=agents,
    rules=rules,
)

task = "Let's create a comprehensive analysis following our rules."
response = group_chat.run(task)
print(response)
```

**Example Output:**

```json
{
    "messages": [
        {
            "role": "System",
            "content": "Group Chat Name: Rules Team\nRules:\n1. Always acknowledge other agents' contributions..."
        },
        {
            "role": "User",
            "content": "Let's create a comprehensive analysis following our rules."
        },
        {
            "role": "analyst",
            "content": "I'll start the analysis. Following our rules, I'll ensure to acknowledge contributions..."
        },
        {
            "role": "researcher",
            "content": "Building on @analyst's initial analysis, I've added research context..."
        }
    ]
}
```

### Timestamps

!!! example "Enabling/Disabling Timestamps"
    Control timestamp generation:

```python
# With timestamps (default)
group_chat_with_time = GroupChat(
    name="Time Team",
    description="A team with timestamps",
    agents=agents,
    time_enabled=True,
)

# Without timestamps
group_chat_no_time = GroupChat(
    name="No Time Team",
    description="A team without timestamps",
    agents=agents,
    time_enabled=False,
)

task = "Let's discuss the project."
response_with_time = group_chat_with_time.run(task)
response_no_time = group_chat_no_time.run(task)

print("With Timestamps:")
print(response_with_time)
print("\nWithout Timestamps:")
print(response_no_time)
```

**Example Output:**

```text
With Timestamps:
{
    "messages": [
        {
            "role": "User",
            "content": "Let's discuss the project.",
            "timestamp": "2024-01-15T10:30:45Z"
        },
        {
            "role": "analyst",
            "content": "Project discussion analysis...",
            "timestamp": "2024-01-15T10:30:47Z"
        }
    ]
}

Without Timestamps:
{
    "messages": [
        {
            "role": "User",
            "content": "Let's discuss the project."
        },
        {
            "role": "analyst",
            "content": "Project discussion analysis..."
        }
    ]
}
```

## Complete Example: Marketing Team

!!! success "Full Implementation"
    A complete example showcasing multiple features:

```python
from dotenv import load_dotenv
import os
from swarms import Agent, GroupChat

load_dotenv()

# Create the marketing team
market_researcher = Agent(
    agent_name="market_researcher",
    system_prompt="You are a market research specialist. You analyze market trends, customer behavior, and competitive landscape.",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=True,
)

content_strategist = Agent(
    agent_name="content_strategist",
    system_prompt="You are a content strategist. You create engaging content strategies and messaging frameworks.",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=True,
)

data_analyst = Agent(
    agent_name="data_analyst",
    system_prompt="You are a data analyst. You analyze campaign performance, metrics, and ROI data.",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=True,
)

creative_director = Agent(
    agent_name="creative_director",
    system_prompt="You are a creative director. You oversee creative vision and ensure brand consistency.",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=True,
)

agents = [
    market_researcher,
    content_strategist,
    data_analyst,
    creative_director,
]

# Initialize GroupChat with dynamic speaker function
marketing_chat = GroupChat(
    name="Marketing Strategy Team",
    description="A collaborative team for developing marketing strategies",
    agents=agents,
    speaker_function="random-dynamic-speaker",
    speaker_state={"strategy": "sequential"},
    max_loops=10,
    output_type="all",
    rules="""
    1. Always acknowledge other team members' contributions
    2. Build upon previous insights
    3. Use @mentions when you need input from specific team members
    4. Clearly state when your part is complete
    """,
)

# Run a complex collaborative task
task = """
@market_researcher Please start by researching the target market for our new product.
Feel free to ask @data_analyst for any data you need.
Once you have insights, @content_strategist can develop the messaging strategy.
@creative_director will ensure brand alignment throughout.
"""

print("Starting marketing strategy session...")
response = marketing_chat.run(task)

print("\n" + "="*80)
print("MARKETING STRATEGY SESSION RESULTS")
print("="*80)
print(response)

# Access conversation history
print("\n" + "="*80)
print("CONVERSATION HISTORY")
print("="*80)
history = marketing_chat.conversation.return_history_as_string()
print(history)
```

**Example Output:**

```text
Starting marketing strategy session...

================================================================================
MARKETING STRATEGY SESSION RESULTS
================================================================================
{
    "messages": [
        {
            "role": "System",
            "content": "Group Chat Name: Marketing Strategy Team\nRules:\n1. Always acknowledge other team members' contributions..."
        },
        {
            "role": "User",
            "content": "@market_researcher Please start by researching the target market..."
        },
        {
            "role": "market_researcher",
            "content": "I've researched the target market. Key findings include demographic trends and competitor analysis. @data_analyst, can you provide sales data to support these findings?"
        },
        {
            "role": "data_analyst",
            "content": "I've reviewed @market_researcher's findings. Here's the supporting sales data: Q1 shows 20% growth in target segments..."
        },
        {
            "role": "market_researcher",
            "content": "Thanks @data_analyst! With that data, the market research is complete. @content_strategist, you can now develop the messaging strategy."
        },
        {
            "role": "content_strategist",
            "content": "Based on @market_researcher's research and @data_analyst's data, I've developed a messaging strategy focused on..."
        },
        {
            "role": "creative_director",
            "content": "I've reviewed all contributions. The messaging aligns with our brand guidelines. The strategy is ready for implementation."
        }
    ],
    "metadata": {
        "name": "Marketing Strategy Team",
        "total_messages": 7
    }
}

================================================================================
CONVERSATION HISTORY
================================================================================
System: Group Chat Name: Marketing Strategy Team...
User: @market_researcher Please start by researching the target market...
market_researcher: I've researched the target market. Key findings include...
data_analyst: I've reviewed @market_researcher's findings. Here's the supporting sales data...
market_researcher: Thanks @data_analyst! With that data, the market research is complete...
content_strategist: Based on @market_researcher's research and @data_analyst's data, I've developed...
creative_director: I've reviewed all contributions. The messaging aligns with our brand guidelines...
```

## Best Practices

### 1. Agent Design

!!! tip "Agent Best Practices"
    - **Clear Roles**: Give each agent a distinct, well-defined role
    - **Descriptive Names**: Use clear, descriptive agent names
    - **Focused Prompts**: Keep system prompts focused on the agent's expertise
    - **Appropriate Models**: Choose models that match the task complexity

### 2. Speaker Function Selection

!!! tip "Speaker Function Guidelines"
    - **Round Robin**: Use for balanced participation
    - **Random**: Use for creative brainstorming
    - **Priority**: Use when certain agents should speak more often
    - **Dynamic**: Use for complex, multi-turn collaborations

### 3. @Mention Usage

!!! tip "@Mention Best Practices"
    - **Be Specific**: Mention only the agents needed for the task
    - **Clear Instructions**: Provide clear instructions when mentioning agents
    - **Allow Delegation**: Let agents mention each other for natural flow
    - **Avoid Over-mentioning**: Don't mention all agents unless necessary

### 4. Conversation Management

!!! tip "Conversation Management"
    - **Set Appropriate max_loops**: Balance between thoroughness and efficiency
    - **Use Rules**: Define clear rules for agent behavior
    - **Monitor History**: Check conversation history for quality
    - **Handle Errors**: Implement error handling for agent failures

### 5. Performance Optimization

!!! tip "Performance Tips"
    - **Use Appropriate Models**: Choose models that match your needs
    - **Limit Context Length**: Set appropriate context lengths
    - **Batch Processing**: Use batched_run for multiple independent tasks
    - **Concurrent Execution**: Use concurrent_run for parallel processing

## Troubleshooting

!!! warning "Common Problems and Solutions"


| Issue Area                     | Common Problem                                                         | Solution/Check                                    |
|------------------------------- |-----------------------------------------------------------------------|---------------------------------------------------|
| API & Setup                    | API keys not working                                                  | Check API keys are set correctly                  |
| Agent Configuration            | Agents not behaving as expected                                       | Verify agent configurations                       |
| Connectivity                   | Errors relating to server/model access                                | Check network connectivity                        |
| Agent Naming                   | Agent not recognized                                                  | Ensure agent names match exactly (case-sensitive) |
| Agent Initialization           | Agent fails to start                                                  | Check that agents are properly initialized        |
| Speaker Function               | Mentions not working                                                  | Verify speaker function supports mentions         |
| Conversation Length            | Runs too long                                                         | Reduce max_loops value                            |
| Conversation Loops             | Infinite or repetitive @mentions                                      | Check for circular @mention patterns              |
| Speaker Function Logic         | Unexpected conversation order                                         | Verify speaker function logic                     |
| Rules & Prompts                | Agents giving poor or generic responses                               | Add more specific rules                           |
| System Prompt                  | Agents stray off-topic                                                | Improve agent system prompts                      |
| Speaker Function Selection     | Suboptimal agent interaction                                          | Use appropriate speaker functions                 |

## Additional Resources

| Resource                     | Description                                      | Link                                                                                   |
|------------------------------|--------------------------------------------------|----------------------------------------------------------------------------------------|
| GroupChat API Reference      | Reference for the GroupChat system               | [View](https://docs.swarms.world/en/latest/swarms/structs/group_chat/)                 |
| GroupChat Guide  | Step-by-step guide for GroupChat     | [View](https://docs.swarms.world/en/latest/swarms/examples/groupchat_comprehensive_examples/)               |
| Agent Documentation          | Reference for building and using agents          | [View](https://docs.swarms.world/en/latest/swarms/structs/agent/)                      |
| Multi-Agent Architectures    | Concepts and architectures for multi-agent swarms| [View](https://docs.swarms.world/en/latest/swarms/concept/swarm_architectures/)        |


## Connect With Us

Join our community for support, updates, and insights:

| Platform | Description | Link |
| --- | --- | --- |
| 📚 Documentation | Official documentation | [docs.swarms.world](https://docs.swarms.world) |
| 💬 Discord | Community support | [Join Discord](https://discord.gg/EamjgSaEQf) |
| 🐦 Twitter | Latest news | [@swarms_corp](https://x.com/swarms_corp) |
| 🚀 GitHub | Source code | [kyegomez/swarms](https://github.com/kyegomez/swarms) |

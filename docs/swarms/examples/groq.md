# Agent with Groq

- Add your `GROQ_API_KEY`

- Initiate your agent

- Run your agent

```python
import os

from swarms import Agent

company = "NVDA"


# Initialize the Managing Director agent
managing_director = Agent(
    agent_name="Managing-Director",
    system_prompt=f"""
    As the Managing Director at Blackstone, your role is to oversee the entire investment analysis process for potential acquisitions. 
    Your responsibilities include:
    1. Setting the overall strategy and direction for the analysis
    2. Coordinating the efforts of the various team members and ensuring a comprehensive evaluation
    3. Reviewing the findings and recommendations from each team member
    4. Making the final decision on whether to proceed with the acquisition
    
    For the current potential acquisition of {company}, direct the tasks for the team to thoroughly analyze all aspects of the company, including its financials, industry position, technology, market potential, and regulatory compliance. Provide guidance and feedback as needed to ensure a rigorous and unbiased assessment.
    """,
    model_name="groq/deepseek-r1-distill-qwen-32b",
    max_loops=1,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    state_save_file_type="json",
    saved_state_path="managing-director.json",
)
```
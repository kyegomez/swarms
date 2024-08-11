import os
from swarms import Agent
from swarms.models.popular_llms import OpenAIChatLLM
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

# This tests uses the same code as the example.py file, with 1 different line


def test_openai_no_quota():
    # Get the OpenAI API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    # Create an instance of the OpenAIChat class
    model = OpenAIChatLLM(
        api_key=api_key,
        model_name="gpt-4o-mini",
        temperature=0.1,
        # This is the only difference, we redirect the request to our test server
        base_url="http://localhost:8000/openai/success",
    )

    # Initialize the agent
    agent = Agent(
        agent_name="Financial-Analysis-Agent",
        system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
        llm=model,
        max_loops=1,
        autosave=True,
        # dynamic_temperature_enabled=True,
        dashboard=False,
        verbose=True,
        # interactive=True, # Set to False to disable interactive mode
        dynamic_temperature_enabled=True,
        saved_state_path="finance_agent.json",
        # tools=[#Add your functions here# ],
        # stopping_token="Stop!",
        # interactive=True,
        # docs_folder="docs", # Enter your folder name
        # pdf_path="docs/finance_agent.pdf",
        # sop="Calculate the profit for a company.",
        # sop_list=["Calculate the profit for a company."],
        user_name="swarms_corp",
        # # docs=
        # # docs_folder="docs",
        retry_attempts=1,
        # context_length=1000,
        # tool_schema = dict
        context_length=200000,
        # tool_schema=
    )

    response = agent.run(
        "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria"
    )
    assert response == "Hello! How can I assist you today?"

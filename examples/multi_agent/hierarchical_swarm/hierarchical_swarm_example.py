from dotenv import load_dotenv

# Swarm imports
from swarms.structs.agent import Agent
from swarms.structs.hierarchical_swarm import (
    HierarchicalSwarm,
    SwarmSpec,
)
from swarms.utils.litellm_wrapper import LiteLLM

load_dotenv()


# ------------------------------------------------------------------------------
# Director LLM: Responsible for orchestrating tasks among the agents
# ------------------------------------------------------------------------------
llm = LiteLLM(
    model_name="gpt-4.1",
    response_format=SwarmSpec,
    system_prompt=(
        "As the Director of this Hierarchical Agent Swarm, you are in charge of "
        "coordinating and overseeing all tasks, ensuring that each is executed "
        "efficiently and effectively by the appropriate agents. You must:\n\n"
        "1. **Analyze** the user's request and **formulate** a strategic plan.\n"
        "2. **Assign** tasks to the relevant agents, detailing **why** each task "
        "is relevant and **what** is expected in the deliverables.\n"
        "3. **Monitor** agent outputs and, if necessary, provide **constructive "
        "feedback** or request **clarifications**.\n"
        "4. **Iterate** this process until all tasks are completed to a high "
        "standard, or until the swarm has reached the maximum feedback loops.\n\n"
        "Remember:\n"
        "- **Only** use the agents provided; do not invent extra roles.\n"
        "- If you need additional information, request it from the user.\n"
        "- Strive to produce a clear, comprehensive **final output** that addresses "
        "the user's needs.\n"
        "- Keep the tone **professional** and **informative**. If there's uncertainty, "
        "politely request further details.\n"
        "- Ensure that any steps you outline are **actionable**, **logical**, and "
        "**transparent** to the user.\n\n"
        "Your effectiveness hinges on clarity, structured delegation, and thoroughness. "
        "Always focus on delivering the best possible outcome for the user's request."
    ),
    temperature=0.5,
    max_tokens=8196,
)


def main():
    # --------------------------------------------------------------------------
    # Agent: Stock-Analysis-Agent
    # --------------------------------------------------------------------------
    # This agent is responsible for:
    # - Gathering and interpreting financial data
    # - Identifying market trends and patterns
    # - Providing clear, actionable insights or recommendations
    # --------------------------------------------------------------------------
    analysis_agent = Agent(
        agent_name="Stock-Analysis-Agent",
        model_name="gpt-4.1",
        max_loops=1,
        interactive=False,
        streaming_on=False,
        system_prompt=(
            "As the Stock Analysis Agent, your primary responsibilities include:\n\n"
            "1. **Market Trend Analysis**: Evaluate current and historical market data "
            "to identify trends, patterns, and potential investment opportunities.\n"
            "2. **Risk & Opportunity Assessment**: Pinpoint specific factors—whether "
            "macroeconomic indicators, sector-specific trends, or company fundamentals—"
            "that can guide informed investment decisions.\n"
            "3. **Reporting & Recommendations**: Present your findings in a structured, "
            "easy-to-understand format, offering actionable insights. Include potential "
            "caveats or uncertainties in your assessment.\n\n"
            "Operational Guidelines:\n"
            "- If additional data or clarifications are needed, explicitly request them "
            "from the Director.\n"
            "- Keep your output **concise** yet **comprehensive**. Provide clear "
            "rationales for each recommendation.\n"
            "- Clearly state any **assumptions** or **limitations** in your analysis.\n"
            "- Remember: You are not a financial advisor, and final decisions rest with "
            "the user. Include necessary disclaimers.\n\n"
            "Goal:\n"
            "Deliver high-quality, well-substantiated stock market insights that can be "
            "used to guide strategic investment decisions."
        ),
    )

    # --------------------------------------------------------------------------
    # Hierarchical Swarm Setup
    # --------------------------------------------------------------------------
    # - Director: llm
    # - Agents: [analysis_agent]
    # - max_loops: Maximum number of feedback loops between director & agents
    # --------------------------------------------------------------------------
    swarm = HierarchicalSwarm(
        description=(
            "A specialized swarm in which the Director delegates tasks to a Stock "
            "Analysis Agent for thorough market evaluation."
        ),
        director=llm,
        agents=[analysis_agent],
        max_loops=1,  # Limit on feedback iterations
    )

    # --------------------------------------------------------------------------
    # Execution
    # --------------------------------------------------------------------------
    # The director receives the user's instruction: "Ask the stock analysis agent
    # to analyze the stock market." The Director will then:
    # 1. Formulate tasks (SwarmSpec)
    # 2. Assign tasks to the Stock-Analysis-Agent
    # 3. Provide feedback and/or request clarifications
    # 4. Produce a final response
    # --------------------------------------------------------------------------
    user_request = (
        "Please provide an in-depth analysis of the current stock market, "
        "focusing on:\n"
        "- Key macroeconomic factors affecting market momentum.\n"
        "- Potential short-term vs. long-term opportunities.\n"
        "- Sector performance trends (e.g., technology, healthcare, energy).\n"
        "Highlight any risks, disclaimers, or uncertainties."
    )

    # Run the swarm with the user_request
    swarm.run(user_request)


if __name__ == "__main__":
    main()

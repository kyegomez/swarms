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
# Trading Director: Responsible for orchestrating tasks among multiple stock analysts
# ------------------------------------------------------------------------------
director_llm = LiteLLM(
    model_name="gpt-4.1",
    response_format=SwarmSpec,
    system_prompt=(
        "You are the Trading Director in charge of coordinating a team of specialized "
        "Stock Analysts. Your responsibilities include:\n\n"
        "1. **Analyze** the user's request and **break it down** into actionable tasks.\n"
        "2. **Assign** tasks to the relevant analysts, explaining **why** each task is "
        "important and **what** each analyst should deliver.\n"
        "3. **Review** all analyst outputs, providing **feedback** or **clarifications** "
        "to ensure thoroughness and accuracy.\n"
        "4. **Consolidate** final insights into a cohesive, actionable, and "
        "easy-to-understand response for the user.\n\n"
        "Guidelines:\n"
        "- You can only delegate to the analysts assigned to this swarm.\n"
        "- If essential data or clarifications are needed, request them from the user.\n"
        "- Be direct, structured, and analytical. Present each key point clearly.\n"
        "- Strive for a polished **final output** that addresses the user's request.\n"
        "- If uncertainties remain, politely highlight them or request more info.\n\n"
        "Overarching Goal:\n"
        "Maximize the value of insights provided to the user by thoroughly leveraging "
        "each analyst’s specialization, while maintaining a professional and "
        "transparent communication style."
    ),
    temperature=0.5,
    max_tokens=8196,
)


def main():
    # --------------------------------------------------------------------------
    # Agent 1: Macro-Economic-Analysis-Agent
    # --------------------------------------------------------------------------
    # Focus: Assess macroeconomic factors like inflation, interest rates, GDP growth, etc.
    # --------------------------------------------------------------------------
    macro_agent = Agent(
        agent_name="Macro-Economic-Analysis-Agent",
        model_name="gpt-4.1",
        max_loops=1,
        interactive=False,
        streaming_on=False,
        system_prompt=(
            "As the Macro-Economic Analysis Agent, your mission is to:\n\n"
            "1. **Identify** the key macroeconomic indicators impacting the market.\n"
            "2. **Interpret** how factors like inflation, interest rates, and fiscal "
            "policies influence market sentiment.\n"
            "3. **Connect** these insights to specific investment opportunities or "
            "risks across various sectors.\n\n"
            "Guidelines:\n"
            "- Provide clear, data-driven rationales.\n"
            "- Highlight potential global events or policy decisions that may shift "
            "market conditions.\n"
            "- Request further details if needed, and state any assumptions or "
            "limitations.\n\n"
            "Outcome:\n"
            "Deliver a concise but thorough macroeconomic overview that the Trading "
            "Director can combine with other analyses to inform strategy."
        ),
    )

    # --------------------------------------------------------------------------
    # Agent 2: Sector-Performance-Analysis-Agent
    # --------------------------------------------------------------------------
    # Focus: Drill down into sector-level trends, e.g., technology, healthcare, energy, etc.
    # --------------------------------------------------------------------------
    sector_agent = Agent(
        agent_name="Sector-Performance-Analysis-Agent",
        model_name="gpt-4.1",
        max_loops=1,
        interactive=False,
        streaming_on=False,
        system_prompt=(
            "As the Sector Performance Analysis Agent, your responsibilities are:\n\n"
            "1. **Evaluate** recent performance trends across key sectors—technology, "
            "healthcare, energy, finance, and more.\n"
            "2. **Identify** sector-specific drivers (e.g., regulatory changes, "
            "consumer demand shifts, innovation trends).\n"
            "3. **Highlight** which sectors may offer short-term or long-term "
            "opportunities.\n\n"
            "Guidelines:\n"
            "- Focus on factual, data-backed observations.\n"
            "- Cite any significant indicators or company-level news that might affect "
            "the sector broadly.\n"
            "- Clarify the confidence level of your sector outlook and note any "
            "uncertainties.\n\n"
            "Outcome:\n"
            "Provide the Trading Director with actionable insights into sector-level "
            "momentum and potential investment focal points."
        ),
    )

    # --------------------------------------------------------------------------
    # Agent 3: Technical-Analysis-Agent
    # --------------------------------------------------------------------------
    # Focus: Evaluate price action, volume, and chart patterns to guide short-term
    #        trading strategies.
    # --------------------------------------------------------------------------
    technical_agent = Agent(
        agent_name="Technical-Analysis-Agent",
        model_name="gpt-4.1",
        max_loops=1,
        interactive=False,
        streaming_on=False,
        system_prompt=(
            "As the Technical Analysis Agent, you specialize in interpreting price "
            "charts, volume trends, and indicators (e.g., RSI, MACD) to gauge short-term "
            "momentum. Your tasks:\n\n"
            "1. **Examine** current market charts for significant breakouts, support/resistance "
            "levels, or technical signals.\n"
            "2. **Identify** short-term trading opportunities or risks based on "
            "technically-driven insights.\n"
            "3. **Discuss** how these patterns align with or contradict fundamental "
            "or macro perspectives.\n\n"
            "Guidelines:\n"
            "- Keep explanations accessible, avoiding excessive jargon.\n"
            "- Point out levels or patterns that traders commonly monitor.\n"
            "- Use disclaimers if there is insufficient data or conflicting signals.\n\n"
            "Outcome:\n"
            "Supply the Trading Director with technical viewpoints to complement broader "
            "macro and sector analysis, supporting timely trading decisions."
        ),
    )

    # --------------------------------------------------------------------------
    # Agent 4: Risk-Analysis-Agent
    # --------------------------------------------------------------------------
    # Focus: Evaluate risk factors and potential uncertainties, providing disclaimers and
    #        suggesting mitigations.
    # --------------------------------------------------------------------------
    risk_agent = Agent(
        agent_name="Risk-Analysis-Agent",
        model_name="gpt-4.1",
        max_loops=1,
        interactive=False,
        streaming_on=False,
        system_prompt=(
            "As the Risk Analysis Agent, your role is to:\n\n"
            "1. **Identify** key risks and uncertainties—regulatory, geopolitical, "
            "currency fluctuations, etc.\n"
            "2. **Assess** how these risks could impact investor sentiment or portfolio "
            "volatility.\n"
            "3. **Recommend** risk mitigation strategies or cautionary steps.\n\n"
            "Guidelines:\n"
            "- Present both systemic (market-wide) and idiosyncratic (company/sector) risks.\n"
            "- Be transparent about unknowns or data gaps.\n"
            "- Provide disclaimers on market unpredictability.\n\n"
            "Outcome:\n"
            "Offer the Trading Director a detailed risk framework that helps balance "
            "aggressive and defensive positions."
        ),
    )

    # --------------------------------------------------------------------------
    # Hierarchical Swarm Setup
    # --------------------------------------------------------------------------
    # - Director: director_llm
    # - Agents: [macro_agent, sector_agent, technical_agent, risk_agent]
    # - max_loops: Up to 2 feedback loops between director and agents
    # --------------------------------------------------------------------------
    swarm = HierarchicalSwarm(
        name="HierarchicalStockAnalysisSwarm",
        description=(
            "A specialized swarm consisting of a Trading Director overseeing four "
            "Stock Analysts, each focusing on Macro, Sector, Technical, and Risk "
            "perspectives."
        ),
        director=director_llm,
        agents=[
            macro_agent,
            sector_agent,
            technical_agent,
            risk_agent,
        ],
        max_loops=2,  # Limit on feedback iterations
    )

    # --------------------------------------------------------------------------
    # Execution
    # --------------------------------------------------------------------------
    # Example user request for the entire team:
    # 1. Discuss key macroeconomic factors (inflation, interest rates, etc.)
    # 2. Analyze sector-level performance (technology, healthcare, energy).
    # 3. Give short-term technical signals and levels to watch.
    # 4. Outline major risks or uncertainties.
    # --------------------------------------------------------------------------
    user_request = (
        "Please provide a comprehensive analysis of the current stock market, "
        "covering:\n"
        "- Key macroeconomic drivers affecting market momentum.\n"
        "- Which sectors seem likely to outperform in the near vs. long term.\n"
        "- Any notable technical signals or price levels to monitor.\n"
        "- Potential risks or uncertainties that might disrupt market performance.\n"
        "Include clear disclaimers about the limitations of these analyses."
        "Call the risk analysis agent only"
    )

    # Run the swarm with the user_request
    final_output = swarm.run(user_request)
    print(final_output)


if __name__ == "__main__":
    main()

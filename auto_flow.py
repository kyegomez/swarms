import os
import json
from pydantic import BaseModel, Field
from swarm_models import OpenAIFunctionCaller
from dotenv import load_dotenv
from typing import Any, List

load_dotenv()


class Flow(BaseModel):
    id: str = Field(
        description="A unique identifier for the flow. This should be a short, descriptive name that captures the main purpose of the flow. Use - to separate words and make it lowercase."
    )
    plan: str = Field(
        description="The comprehensive plan detailing how the flow will accomplish the given task. This should include the high-level strategy, key milestones, and expected outcomes. The plan should clearly articulate what the overall goal is, what success looks like, and how progress will be measured throughout execution."
    )
    failures_prediction: str = Field(
        description="A thorough analysis of potential failure modes and mitigation strategies. This should identify technical risks, edge cases, error conditions, and possible points of failure in the flow. For each identified risk, include specific preventive measures, fallback approaches, and recovery procedures to ensure robustness and reliability."
    )
    rationale: str = Field(
        description="The detailed reasoning and justification for why this specific flow design is optimal for the given task. This should explain the key architectural decisions, tradeoffs considered, alternatives evaluated, and why this approach best satisfies the requirements. Include both technical and business factors that influenced the design."
    )
    flow: str = Field(
        description="The precise execution flow defining how agents interact and coordinate. Use -> to indicate sequential processing where one agent must complete before the next begins (e.g. agent1 -> agent2 -> agent3). Use , to indicate parallel execution where multiple agents can run simultaneously (e.g. agent1 -> agent2, agent3, agent4). The flow should clearly show the dependencies and parallelization opportunities between agents. You must only use the agent names provided in the task description do not make up new agent names and do not use any other formatting."
    )


class AgentRearrangeBuilder(BaseModel):
    name: str = Field(
        description="The name of the swarm. This should be a short, descriptive name that captures the main purpose of the flow."
    )
    description: str = Field(
        description="A brief description of the swarm. This should be a concise summary of the main purpose of the swarm."
    )
    flows: List[Flow] = Field(
        description="A list of flows that are optimal for the given task. Each flow should be a detailed plan, failure prediction, rationale, and execution flow."
    )
    swarm_flow: str = Field(
        description="The flow defining how each team should communicate and coordinate with eachother.Use -> to indicate sequential processing where one id must complete before the next begins (e.g. team1 -> team2 -> team3). Use , to indicate parallel execution where multiple teams can run simultaneously (e.g. team1 -> team2, team3, team4). The flow should clearly show the dependencies and parallelization opportunities between teams. You must only use the team names provided in the id do not make up new team names and do not use any other formatting."
    )


# def flow_generator(task: str) -> Flow:


def setup_model(base_model: BaseModel = Flow):
    model = OpenAIFunctionCaller(
        system_prompt="""You are an expert flow architect specializing in designing multi-agent workflows. Your role is to analyze tasks and create optimal execution flows that coordinate multiple AI agents effectively.

        When given a task, you will:
        1. Develop a comprehensive plan breaking down the task into logical steps
        2. Carefully consider potential failure modes and build in robust error handling
        3. Provide clear rationale for your architectural decisions and agent coordination strategy
        4. Design a precise flow showing both sequential dependencies and parallel execution opportunities

        Your flows should maximize:
        - Efficiency through smart parallelization
        - Reliability through thorough error handling
        - Clarity through well-structured agent interactions
        - Effectiveness through strategic task decomposition

        Format your flow using -> for sequential steps and , for parallel execution. Be specific about agent roles and interactions.
        """,
        base_model=base_model,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.5,
    )
    return model


def generate_flow(task: str) -> Any:
    model = setup_model()
    flow = model.run(task)
    print(json.dumps(flow, indent=4))
    return flow


def generate_agent_rearrange(task: str) -> Any:
    model = setup_model(base_model=AgentRearrangeBuilder)
    flow = model.run(task)
    print(json.dumps(flow, indent=4))
    return flow


if __name__ == "__main__":
    # Basic patient diagnosis flow
    # generate_flow("Diagnose a patient's symptoms and create a treatment plan. You have 3 agents to use: Diagnostician, Specialist, CareCoordinator")

    # # Complex multi-condition case
    # generate_flow("""Handle a complex patient case with multiple chronic conditions requiring ongoing care coordination.
    #              The patient has diabetes, heart disease, and chronic pain.
    #              Create a comprehensive diagnosis and treatment plan.
    #              You have 3 agents to use: Diagnostician, Specialist, CareCoordinator""")

    # # Emergency trauma case
    # generate_flow("""Process an emergency trauma case requiring rapid diagnosis and immediate intervention.
    #              Patient presents with multiple injuries from a car accident.
    #              Develop immediate and long-term treatment plans.
    #              You have 3 agents to use: Diagnostician, Specialist, CareCoordinator""")

    # # Long-term care planning
    # generate_flow("""Design a 6-month care plan for an elderly patient with declining cognitive function.
    #              Include regular assessments, specialist consultations, and family coordination.
    #              You have 3 agents to use: Diagnostician, Specialist, CareCoordinator""")

    # # Mental health assessment
    # generate_flow("""Conduct a comprehensive mental health assessment and develop treatment strategy.
    #              Patient shows signs of depression and anxiety with possible underlying conditions.
    #              Create both immediate intervention and long-term support plans.
    #              You have 3 agents to use: Diagnostician, Specialist, CareCoordinator""")

    generate_agent_rearrange(
        """Build a complete automated hedge fund system.
                 Design and implement a sophisticated trading strategy incorporating multiple asset classes,
                 risk management protocols, and automated execution systems.
                 The system should include:
                 - Market analysis and research capabilities
                 - Portfolio optimization and risk management
                 - Automated trade execution and settlement
                 - Compliance and regulatory monitoring
                 - Performance tracking and reporting
                 - Fund operations and administration
                 Create a comprehensive architecture that integrates all these components into a fully automated system."""
    )

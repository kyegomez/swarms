"""
Implementation of the MALT (Multi-Agent Learning Task) orchestrator.

MALT is a multi-agent system that orchestrates the interaction between multiple agents
to perform complex tasks through a structured conversation. It ensures reliability and
provides options for output formatting.


- Paper: https://arxiv.org/pdf/2412.01928


Potential Improvements:
- Autonomously create the agents based on the task.
- Feed verifier responses back into the creator to improve the proof.
- Feed refiner responses back into the creator to improve the proof.
- Feed majority voting responses back into the creator to improve the proof.


This is a simplified implementation of the MALT orchestrator. The original implementation trains the models with dpo and sft.
Whereas this implementation uses the models as is.

"""

from typing import List

from loguru import logger

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.multi_agent_exec import run_agents_concurrently
from swarms.utils.any_to_str import any_to_str

# Agent 1: Proof Creator Agent
proof_creator_prompt = """
You are a world-renowned mathematician with an extensive background in multiple advanced fields including number theory, abstract algebra, topology, advanced calculus, and mathematical logic. You are tasked with generating an original, non-trivial theorem along with a fully rigorous and detailed proof. Your response must include the following elements:

1. **Theorem Statement and Definitions:**
   - Clearly articulate a novel theorem in a specific branch of mathematics.
   - Provide precise definitions of any non-standard terms or constructs that appear in your theorem.
   - Contextualize the theorem within the framework of existing mathematical theory, explaining its significance.

2. **Structured Proof:**
   - Develop the proof in a step-by-step format that is logically coherent and rigorous.
   - Include intermediate results such as lemmas, propositions, and corollaries where applicable.
   - Provide thorough justifications for every logical step and transition, referencing known theorems or axioms when relevant.
   - If multiple cases or conditions are involved, handle each case separately with clear demarcations.

3. **Intuitive Explanations and Motivation:**
   - Supplement the formal proof with intuitive explanations that help the reader understand the underlying ideas.
   - Explain why the theorem is interesting, including possible applications or implications in other areas of mathematics.
   - Address potential counterexamples or special conditions that could challenge the generality of your result.

4. **Formatting and Detail:**
   - Your output should be verbose, ensuring that each part of the proof is elaborated in detail.
   - Use formal mathematical language but also include lay explanations for complex arguments.
   - Ensure that the final document is self-contained, clear, and can be understood by both experts and advanced students.

Your response should be as comprehensive as possible, leaving no room for ambiguity, and it should reflect your mastery in constructing original mathematical arguments.
"""

proof_creator_agent = Agent(
    agent_name="Proof-Creator-Agent",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt=proof_creator_prompt,
)

# Agent 2: Proof Verifier Agent
proof_verifier_prompt = """
You are an esteemed mathematician and veteran academic known for your precise and critical evaluations of complex mathematical arguments. Your role is to verify the proof produced by the Proof-Creator-Agent. Your detailed analysis should include the following components:

1. **Comprehensive Logical Analysis:**
   - Examine every logical step of the proof, ensuring that all transitions are valid and that no step is assumed without proper justification.
   - Identify any potential errors, missing justifications, or logical gaps in the argument.
   - Provide a thorough commentary on each lemma, proposition, and conclusion presented in the proof.

2. **Mathematical Rigor and Consistency:**
   - Cross-reference every argument with established mathematical theories, axioms, and known results.
   - Check the consistency of definitions and ensure that they are used uniformly throughout the proof.
   - Address any inconsistencies or ambiguities in notation, assumptions, or logical structure.

3. **Critical Feedback and Suggestions:**
   - Provide detailed feedback on the strengths and weaknesses of the proof.
   - Suggest specific modifications or additional explanations that could enhance the clarity, correctness, and overall rigor.
   - If applicable, propose alternative approaches or insights that could further solidify the argument.

4. **Exhaustive Review:**
   - Your analysis should be extensive and methodical, examining the proof from multiple angles.
   - Ensure that each critique is accompanied by a clear rationale and reference to relevant mathematical principles.
   - Summarize your findings in a structured format, highlighting both the successful aspects of the proof and areas that need improvement.

Your review must be exhaustive, ensuring that even the most subtle aspects of the proof are scrutinized in depth.
"""

proof_verifier_agent = Agent(
    agent_name="Proof-Verifier-Agent",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt=proof_verifier_prompt,
)

# Agent 3: Proof Refiner Agent
proof_refiner_prompt = """
You are an expert in mathematical exposition and refinement with decades of experience in teaching, publishing, and peer-reviewing advanced mathematics. Your objective is to take the initial proof and the comprehensive feedback provided by the Proof-Verifier-Agent, and then produce a refined, polished version of the proof. Your refined output must address the following points:

1. **Incorporation of Verification Feedback:**
   - Meticulously integrate all the detailed suggestions and critiques provided by the Proof-Verifier-Agent.
   - Ensure that all logical gaps, ambiguities, and inconsistencies identified in the review are resolved in the revised proof.
   - Revisit and revise definitions, lemmas, and intermediate steps where necessary to ensure complete logical consistency.

2. **Enhanced Clarity and Structure:**
   - Reorganize the proof for optimal flow and readability, ensuring that each section leads naturally to the next.
   - Add comprehensive explanations where needed, emphasizing intuitive reasoning alongside formal arguments.
   - Break down complex sections into more manageable parts, and ensure that each is clearly labeled and explained.

3. **Rigorous Detailing and Presentation:**
   - Enhance the overall presentation of the proof by ensuring that every assertion is supported by detailed justifications.
   - Include additional commentary that not only defends the logical integrity of the argument but also explains its broader significance.
   - Maintain a balance between rigorous formalism and accessible exposition so that the refined proof appeals to both experts and advanced learners.

4. **Comprehensive Feedback and Rationale:**
   - For every modification made, provide an accompanying explanation that outlines the rationale behind the change.
   - If any aspects of the original proof were retained, clarify why they were considered adequate and how they contribute to the overall argument.
   - Ensure that your final output is a cohesive, self-contained document that stands up to critical academic scrutiny.

Your refined proof should be a masterpiece of mathematical writing, addressing all the feedback with detailed revisions and explanations.
"""

proof_refiner_agent = Agent(
    agent_name="Proof-Refiner-Agent",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt=proof_refiner_prompt,
)


majority_voting_prompt = """
Engage in a comprehensive and exhaustive majority voting analysis of the following conversation, ensuring a deep and thoughtful examination of the responses provided by each agent. This analysis should not only summarize the responses but also critically engage with the content, context, and implications of each agent's input.

Please adhere to the following detailed guidelines:

1. **Identification of Dominant Responses:**
   - Identify the most prevalent answer or recommendation across all agents. Provide a thorough rationale for its dominance, including an exploration of the factors that may have contributed to its acceptance among the agents. Discuss the context in which this consensus emerged and any relevant historical or theoretical frameworks that support this conclusion.

2. **Exploration of Disparities:**
   - Delve into any significant disparities or contrasting viewpoints between agents. Explore the underlying reasons for these differences, considering aspects such as differing methodologies, assumptions, or interpretations of the task at hand. Analyze how these contrasting perspectives may reflect broader debates within the field and what implications they hold for the overall understanding of the topic.

3. **Consensus and Disagreement Analysis:**
   - Highlight key areas of consensus and disagreement among the agents. Discuss the implications of these findings on the overall argument, including how consensus can strengthen certain claims while disagreement may indicate areas of uncertainty or contention. Provide examples from the conversation to illustrate these points and consider how they might influence future discussions or research directions.

4. **Critical Evaluation of Majority Opinion:**
   - Critically evaluate the strength of the majority opinion, considering factors such as the reasoning behind it and its mathematical validity if applicable. Assess whether the majority opinion is well-supported by evidence and logical reasoning, and discuss any potential weaknesses or oversights that may undermine its credibility. 

5. **Insights from Minority Viewpoints:**
   - Note any unique insights from minority viewpoints, assessing their potential contributions to a more nuanced understanding of the topic. Discuss how these minority perspectives can enrich the conversation and provide alternative angles that may have been overlooked by the majority. Consider the value of dissent in academic discourse and how it can lead to more robust conclusions.

6. **Synthesis of Recommendations:**
   - Provide a final synthesized recommendation based on the majority consensus, ensuring that it reflects a thorough consideration of all perspectives and is grounded in sound reasoning. This recommendation should not only summarize the majority view but also integrate insights from minority opinions, creating a comprehensive and balanced conclusion that acknowledges the complexity of the discussion.

Throughout your analysis, focus on uncovering clear patterns while being attentive to the subtleties and complexities inherent in the responses. Pay particular attention to the nuances of mathematical contexts where algorithmic thinking may be required, ensuring that your examination is both rigorous and accessible to a diverse audience.
"""

majority_voting_agent = Agent(
    agent_name="Majority-Voting-Agent",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt=majority_voting_prompt,
)


class MALT:
    """
    MALT (Mult-Agent Learning Task) orchestrates the interaction between multiple agents
    to perform complex tasks through a structured conversation. It ensures reliability and
    provides options for output formatting.

    Attributes:
        main_agent (Agent): The primary agent responsible for executing the main task.
        refiner_agent (Agent): The agent that refines the output of the main agent.
        verifier_agent (Agent): The agent that verifies the output of the main agent.
        max_loops (int): The maximum number of iterations for the task execution.
        return_list (bool): Flag to return output as a list.
        return_dict (bool): Flag to return output as a dictionary.
        agents (list[Agent]): A list of agents to be used in the task.
        conversation (Conversation): Manages the conversation history between agents.
    """

    def __init__(
        self,
        main_agent: Agent = None,
        refiner_agent: Agent = None,
        verifier_agent: Agent = None,
        max_loops: int = 1,
        return_list: bool = False,
        return_dict: bool = False,
        agents: list[Agent] = [],
        preset_agents: bool = True,
    ):
        logger.info(
            "Initializing MALT with provided agents and parameters."
        )
        self.main_agent = main_agent
        self.refiner_agent = refiner_agent
        self.verifier_agent = verifier_agent
        self.max_loops = max_loops
        self.return_list = return_list
        self.return_dict = return_dict
        self.agents = agents

        self.conversation = Conversation()
        logger.debug("Conversation initialized.")

        if preset_agents:
            self.main_agent = proof_creator_agent
            self.refiner_agent = proof_refiner_agent
            self.verifier_agent = proof_verifier_agent

        self.reliability_check()

    def reliability_check(self):
        """Checks the reliability of the provided agents and parameters."""
        logger.info("Performing reliability check.")
        if self.max_loops == 0 or self.max_loops is None:
            logger.error("max_loops must be greater than 0")
            raise ValueError("max_loops must be greater than 0")

        # Check if agents list is provided and not empty when needed
        if not self.agents and (
            self.main_agent is None
            or self.refiner_agent is None
            or self.verifier_agent is None
        ):
            logger.error(
                "Missing agents: Provide individual agents or a list of at least 3 agents."
            )
            raise ValueError(
                "Either provide individual agents (main_agent, refiner_agent, verifier_agent) or a list of at least 3 agents"
            )

        # If individual agents aren't specified but we have agents list, use the first three
        if (
            self.main_agent is None
            or self.refiner_agent is None
            or self.verifier_agent is None
        ) and len(self.agents) >= 3:
            self.main_agent = self.main_agent or self.agents[0]
            self.refiner_agent = self.refiner_agent or self.agents[1]
            self.verifier_agent = (
                self.verifier_agent or self.agents[2]
            )

        # Final check to ensure we have all required agents
        if (
            self.main_agent is None
            or self.refiner_agent is None
            or self.verifier_agent is None
        ):
            logger.error("Missing required agents.")
            raise ValueError(
                "Missing required agents: main_agent, refiner_agent, and verifier_agent must all be provided"
            )
        logger.info("Reliability check passed.")

    def step(self, task: str, img: str = None, *args, **kwargs):
        """Executes the task using the main agent and processes the output through verifier and refiner agents.

        Args:
            task (str): The task to be executed by the main agent.
            img (str, optional): An optional image input for the agents.

        Returns:
            str or list or dict: The output from the conversation based on the specified return format.
        """
        self.conversation.add(
            role="user",
            content=task,
        )

        logger.info("Running task with main agent.")
        main_agent_output = self.main_agent.run(
            task=task, img=img, *args, **kwargs
        )

        self.conversation.add(
            role=self.main_agent.agent_name, content=main_agent_output
        )

        logger.info("Running task with verifier agents")
        verified_outputs = run_agents_concurrently(
            [
                self.verifier_agent,
                self.verifier_agent,
                self.verifier_agent,
            ],
            task=main_agent_output,
            max_workers=3,
        )

        self.conversation.add(
            role=self.verifier_agent.agent_name,
            content=verified_outputs,
        )

        ######################### MAJORITY VOTING #########################

        # Majority Voting on the verified outputs
        majority_voting_verified = majority_voting_agent.run(
            task=any_to_str(verified_outputs),
        )

        self.conversation.add(
            role=majority_voting_agent.agent_name,
            content=majority_voting_verified,
        )

        #########################################################

        # Refining the majority voting output
        logger.info("Running task with refiner agents")
        for output in verified_outputs:
            refined_outputs = run_agents_concurrently(
                [
                    self.refiner_agent,
                    self.refiner_agent,
                    self.refiner_agent,
                ],
                task=output,
                max_workers=3,
            )
            logger.debug(f"Refined outputs: {refined_outputs}")

            self.conversation.add(
                role=self.refiner_agent.agent_name,
                content=refined_outputs,
            )

        return self.conversation.get_str()

    def run(self, task: str, img: str = None, *args, **kwargs):
        """Executes the task using the main agent and processes the output through verifier and refiner agents.

        Args:
            task (str): The task to be executed by the main agent.
            img (str, optional): An optional image input for the agents.

        Returns:
            str or list or dict: The output from the conversation based on the specified return format.
        """
        task = task

        for i in range(self.max_loops):
            logger.info(f"Starting iteration {i+1}/{self.max_loops}")
            output = self.step(task, img, *args, **kwargs)
            if output is not None:
                return output

        if self.return_list:
            logger.info("Returning output as a list.")
            return self.conversation.return_messages_as_list()
        elif self.return_dict:
            logger.info("Returning output as a dictionary.")
            return self.conversation.return_messages_as_dictionary()
        else:
            logger.info("Returning output as a string.")
            return self.conversation.get_str()

    def run_batched(self, tasks: List[str], *args, **kwargs):
        """Executes a list of tasks using the main agent and processes the output through verifier and refiner agents.

        Args:
            tasks (list[str]): The list of tasks to be executed by the main agent.
        """
        logger.info("Running batch of tasks.")
        logger.info(f"Number of tasks: {len(tasks)}")

        outputs = []
        for task in tasks:
            outputs.append(self.run(task, *args, **kwargs))
        return outputs

    def __call__(self, task: str, *args, **kwargs):
        return self.run(task, *args, **kwargs)

    def __str__(self):
        return self.conversation.get_str()

    def __repr__(self):
        return self.conversation.get_str()

"""
LLM Council - A Swarms implementation inspired by Andrej Karpathy's llm-council.

This implementation creates a council of specialized LLM agents that:
1. Each agent responds to the user query independently
2. All agents review and rank each other's (anonymized) responses
3. A Chairman LLM synthesizes all responses and rankings into a final answer

The council demonstrates how different models evaluate and rank each other's work,
often selecting responses from other models as superior to their own.
"""

from typing import Dict, List, Optional
import random
from swarms import Agent
from swarms.structs.multi_agent_exec import (
    run_agents_concurrently,
    batched_grid_agent_execution,
)


def get_gpt_councilor_prompt() -> str:
    """
    Get system prompt for GPT-5.1 councilor.
    
    Returns:
        System prompt string for GPT-5.1 councilor agent.
    """
    return """You are a member of the LLM Council, representing GPT-5.1. Your role is to provide comprehensive, analytical, and thorough responses to user queries.

Your strengths:
- Deep analytical thinking and comprehensive coverage
- Ability to break down complex topics into detailed components
- Thorough exploration of multiple perspectives
- Rich contextual understanding

Your approach:
- Provide detailed, well-structured responses
- Include relevant context and background information
- Consider multiple angles and perspectives
- Be thorough but clear in your explanations

Remember: You are part of a council where multiple AI models will respond to the same query, and then evaluate each other's responses. Focus on quality, depth, and clarity."""


def get_gemini_councilor_prompt() -> str:
    """
    Get system prompt for Gemini 3 Pro councilor.
    
    Returns:
        System prompt string for Gemini 3 Pro councilor agent.
    """
    return """You are a member of the LLM Council, representing Gemini 3 Pro. Your role is to provide concise, well-processed, and structured responses to user queries.

Your strengths:
- Clear and structured communication
- Efficient information processing
- Condensed yet comprehensive responses
- Well-organized presentation

Your approach:
- Provide concise but complete answers
- Structure information clearly and logically
- Focus on key points without unnecessary verbosity
- Present information in an easily digestible format

Remember: You are part of a council where multiple AI models will respond to the same query, and then evaluate each other's responses. Focus on clarity, structure, and efficiency."""


def get_claude_councilor_prompt() -> str:
    """
    Get system prompt for Claude Sonnet 4.5 councilor.
    
    Returns:
        System prompt string for Claude Sonnet 4.5 councilor agent.
    """
    return """You are a member of the LLM Council, representing Claude Sonnet 4.5. Your role is to provide thoughtful, balanced, and nuanced responses to user queries.

Your strengths:
- Nuanced understanding and balanced perspectives
- Thoughtful consideration of trade-offs
- Clear reasoning and logical structure
- Ethical and responsible analysis

Your approach:
- Provide balanced, well-reasoned responses
- Consider multiple viewpoints and implications
- Be thoughtful about potential limitations or edge cases
- Maintain clarity while showing depth of thought

Remember: You are part of a council where multiple AI models will respond to the same query, and then evaluate each other's responses. Focus on thoughtfulness, balance, and nuanced reasoning."""


def get_grok_councilor_prompt() -> str:
    """
    Get system prompt for Grok-4 councilor.
    
    Returns:
        System prompt string for Grok-4 councilor agent.
    """
    return """You are a member of the LLM Council, representing Grok-4. Your role is to provide creative, innovative, and unique perspectives on user queries.

Your strengths:
- Creative problem-solving and innovative thinking
- Unique perspectives and out-of-the-box approaches
- Engaging and dynamic communication style
- Ability to connect seemingly unrelated concepts

Your approach:
- Provide creative and innovative responses
- Offer unique perspectives and fresh insights
- Be engaging and dynamic in your communication
- Think creatively while maintaining accuracy

Remember: You are part of a council where multiple AI models will respond to the same query, and then evaluate each other's responses. Focus on creativity, innovation, and unique insights."""


def get_chairman_prompt() -> str:
    """
    Get system prompt for the Chairman agent.
    
    Returns:
        System prompt string for the Chairman agent.
    """
    return """You are the Chairman of the LLM Council. Your role is to synthesize responses from all council members along with their evaluations and rankings into a final, comprehensive answer.

Your responsibilities:
1. Review all council member responses to the user's query
2. Consider the rankings and evaluations provided by each council member
3. Synthesize the best elements from all responses
4. Create a final, comprehensive answer that incorporates the strengths of different approaches
5. Provide transparency about which perspectives influenced the final answer

Your approach:
- Synthesize rather than simply aggregate
- Identify the strongest elements from each response
- Create a cohesive final answer that benefits from multiple perspectives
- Acknowledge the diversity of approaches taken by council members
- Provide a balanced, comprehensive response that serves the user's needs

Remember: You have access to all original responses and all evaluations. Use this rich context to create the best possible final answer."""


def get_evaluation_prompt(query: str, responses: Dict[str, str], evaluator_name: str) -> str:
    """
    Create evaluation prompt for council members to review and rank responses.
    
    Args:
        query: The original user query
        responses: Dictionary mapping anonymous IDs to response texts
        evaluator_name: Name of the agent doing the evaluation
        
    Returns:
        Formatted evaluation prompt string
    """
    responses_text = "\n\n".join([
        f"Response {response_id}:\n{response_text}"
        for response_id, response_text in responses.items()
    ])
    
    return f"""You are evaluating responses from your fellow LLM Council members to the following query:

QUERY: {query}

Below are the anonymized responses from all council members (including potentially your own):

{responses_text}

Your task:
1. Carefully read and analyze each response
2. Evaluate the quality, accuracy, completeness, and usefulness of each response
3. Rank the responses from best to worst (1 = best, {len(responses)} = worst)
4. Provide brief reasoning for your rankings
5. Be honest and objective - you may find another model's response superior to your own

Format your evaluation as follows:

RANKINGS:
1. Response [ID]: [Brief reason why this is the best]
2. Response [ID]: [Brief reason]
...
{len(responses)}. Response [ID]: [Brief reason why this ranks lowest]

ADDITIONAL OBSERVATIONS:
[Any additional insights about the responses, common themes, strengths/weaknesses, etc.]

Remember: The goal is honest, objective evaluation. If another model's response is genuinely better, acknowledge it."""


def get_synthesis_prompt(
    query: str,
    original_responses: Dict[str, str],
    evaluations: Dict[str, str],
    id_to_member: Dict[str, str]
) -> str:
    """
    Create synthesis prompt for the Chairman.
    
    Args:
        query: Original user query
        original_responses: Dict mapping member names to their responses
        evaluations: Dict mapping evaluator names to their evaluation texts
        id_to_member: Mapping from anonymous IDs to member names
        
    Returns:
        Formatted synthesis prompt
    """
    responses_section = "\n\n".join([
        f"=== {name} ===\n{response}"
        for name, response in original_responses.items()
    ])
    
    evaluations_section = "\n\n".join([
        f"=== Evaluation by {name} ===\n{evaluation}"
        for name, evaluation in evaluations.items()
    ])
    
    return f"""As the Chairman of the LLM Council, synthesize the following information into a final, comprehensive answer.

ORIGINAL QUERY:
{query}

COUNCIL MEMBER RESPONSES:
{responses_section}

COUNCIL MEMBER EVALUATIONS AND RANKINGS:
{evaluations_section}

ANONYMOUS ID MAPPING (for reference):
{chr(10).join([f"  {aid} = {name}" for aid, name in id_to_member.items()])}

Your task:
1. Review all council member responses
2. Consider the evaluations and rankings provided by each member
3. Identify the strongest elements from each response
4. Synthesize a final, comprehensive answer that:
   - Incorporates the best insights from multiple perspectives
   - Addresses the query thoroughly and accurately
   - Benefits from the diversity of approaches taken
   - Is clear, well-structured, and useful

Provide your final synthesized response below. You may reference which perspectives or approaches influenced different parts of your answer."""


class LLMCouncil:
    """
    An LLM Council that orchestrates multiple specialized agents to collaboratively
    answer queries through independent responses, peer review, and synthesis.
    
    The council follows this workflow:
    1. Dispatch query to all council members in parallel
    2. Collect all responses (anonymized)
    3. Have each member review and rank all responses
    4. Chairman synthesizes everything into final response
    """
    
    def __init__(
        self,
        council_members: Optional[List[Agent]] = None,
        chairman_model: str = "gpt-5.1",
        verbose: bool = True,
    ):
        """
        Initialize the LLM Council.
        
        Args:
            council_members: List of Agent instances representing council members.
                           If None, creates default council with GPT-5.1, Gemini 3 Pro,
                           Claude Sonnet 4.5, and Grok-4.
            chairman_model: Model name for the Chairman agent that synthesizes responses.
            verbose: Whether to print progress and intermediate results.
        """
        self.verbose = verbose
        
        # Create default council members if none provided
        if council_members is None:
            self.council_members = self._create_default_council()
        else:
            self.council_members = council_members
        
        # Create Chairman agent
        self.chairman = Agent(
            agent_name="Chairman",
            agent_description="Chairman of the LLM Council, responsible for synthesizing all responses and rankings into a final answer",
            system_prompt=get_chairman_prompt(),
            model_name=chairman_model,
            max_loops=1,
            verbose=verbose,
            temperature=0.7,
        )
        
        if self.verbose:
            print(f"🏛️  LLM Council initialized with {len(self.council_members)} members")
            for i, member in enumerate(self.council_members, 1):
                print(f"   {i}. {member.agent_name} ({member.model_name})")
    
    def _create_default_council(self) -> List[Agent]:
        """
        Create default council members with specialized prompts and models.
        
        Returns:
            List of Agent instances configured as council members.
        """
        
        # GPT-5.1 Agent - Analytical and comprehensive
        gpt_agent = Agent(
            agent_name="GPT-5.1-Councilor",
            agent_description="Analytical and comprehensive AI councilor specializing in deep analysis and thorough responses",
            system_prompt=get_gpt_councilor_prompt(),
            model_name="gpt-5.1",
            max_loops=1,
            verbose=False,
            temperature=0.7,
        )
        
        # Gemini 3 Pro Agent - Concise and processed
        gemini_agent = Agent(
            agent_name="Gemini-3-Pro-Councilor",
            agent_description="Concise and well-processed AI councilor specializing in clear, structured responses",
            system_prompt=get_gemini_councilor_prompt(),
            model_name="gemini-2.5-flash",  # Using available Gemini model
            max_loops=1,
            verbose=False,
            temperature=0.7,
        )
        
        # Claude Sonnet 4.5 Agent - Balanced and thoughtful
        claude_agent = Agent(
            agent_name="Claude-Sonnet-4.5-Councilor",
            agent_description="Thoughtful and balanced AI councilor specializing in nuanced and well-reasoned responses",
            system_prompt=get_claude_councilor_prompt(),
            model_name="anthropic/claude-sonnet-4-5",  # Using available Claude model
            max_loops=1,
            verbose=False,
            temperature=0.0,
            top_p=None,
        )
        
        # Grok-4 Agent - Creative and innovative
        grok_agent = Agent(
            agent_name="Grok-4-Councilor",
            agent_description="Creative and innovative AI councilor specializing in unique perspectives and creative solutions",
            system_prompt=get_grok_councilor_prompt(),
            model_name="x-ai/grok-4",  # Using available model as proxy for Grok-4
            max_loops=1,
            verbose=False,
            temperature=0.8,
        )
        
        members = [gpt_agent, gemini_agent, claude_agent, grok_agent]
        
        return members
    
    def run(self, query: str) -> Dict:
        """
        Execute the full LLM Council workflow.
        
        Args:
            query: The user's query to process
            
        Returns:
            Dictionary containing:
                - original_responses: Dict mapping member names to their responses
                - evaluations: Dict mapping evaluator names to their rankings
                - final_response: The Chairman's synthesized final answer
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("🏛️  LLM COUNCIL SESSION")
            print("="*80)
            print(f"\n📝 Query: {query}\n")
        
        # Step 1: Get responses from all council members in parallel
        if self.verbose:
            print("📤 Dispatching query to all council members...")
        
        results_dict = run_agents_concurrently(
            self.council_members,
            task=query,
            return_agent_output_dict=True
        )
        
        # Map results to member names
        original_responses = {
            member.agent_name: response
            for member, response in zip(self.council_members, 
                                       [results_dict.get(member.agent_name, "") 
                                        for member in self.council_members])
        }
        
        if self.verbose:
            print(f"✅ Received {len(original_responses)} responses\n")
            for name, response in original_responses.items():
                print(f"   {name}: {response[:100]}...")
        
        # Step 2: Anonymize responses for evaluation
        # Create anonymous IDs (A, B, C, D, etc.)
        anonymous_ids = [chr(65 + i) for i in range(len(self.council_members))]
        random.shuffle(anonymous_ids)  # Shuffle to ensure anonymity
        
        anonymous_responses = {
            anonymous_ids[i]: original_responses[member.agent_name]
            for i, member in enumerate(self.council_members)
        }
        
        # Create mapping from anonymous ID to member name (for later reference)
        id_to_member = {
            anonymous_ids[i]: member.agent_name
            for i, member in enumerate(self.council_members)
        }
        
        if self.verbose:
            print("\n🔍 Council members evaluating each other's responses...")
        
        # Step 3: Have each member evaluate and rank all responses concurrently
        # Create evaluation tasks for each member
        evaluation_tasks = [
            get_evaluation_prompt(query, anonymous_responses, member.agent_name)
            for member in self.council_members
        ]
        
        # Run evaluations concurrently using batched_grid_agent_execution
        evaluation_results = batched_grid_agent_execution(
            self.council_members,
            evaluation_tasks
        )
        
        # Map results to member names
        evaluations = {
            member.agent_name: evaluation_results[i]
            for i, member in enumerate(self.council_members)
        }
        
        if self.verbose:
            print(f"✅ Received {len(evaluations)} evaluations\n")
        
        # Step 4: Chairman synthesizes everything
        if self.verbose:
            print("👔 Chairman synthesizing final response...\n")
        
        synthesis_prompt = get_synthesis_prompt(
            query, original_responses, evaluations, id_to_member
        )
        
        final_response = self.chairman.run(task=synthesis_prompt)
        
        if self.verbose:
            print(f"{'='*80}")
            print("✅ FINAL RESPONSE")
            print(f"{'='*80}\n")
        
        return {
            "query": query,
            "original_responses": original_responses,
            "evaluations": evaluations,
            "final_response": final_response,
            "anonymous_mapping": id_to_member,
        }


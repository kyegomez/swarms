import os
import time
import json
import asyncio
import uuid
from typing import Dict, List, Optional, Union, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

# 使用现有的Swarms组件
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.formatter import formatter
from swarms.utils.any_to_str import any_to_str
from swarms.utils.litellm_tokenizer import count_tokens

class ElectionSwarm:
    """
    ElectionSwarm implements a voting mechanism across multiple agents to make decisions.
    
    This class manages multiple agents, runs them in parallel on the same task,
    and determines the final output based on voting rules.
    
    Attributes:
        agents (List[Agent]): List of agent instances that participate in the voting
        voting_method (str): Method to use for determining the winning output
        quorum_percentage (float): Percentage of agents required for a majority
        max_voting_rounds (int): Maximum number of voting rounds before fallback
        tie_breaking_method (str): Method to resolve ties
        output_type (str): Format of the output
        election_history (List[Dict]): History of all elections held
        short_memory (Conversation): Memory to store the election process
        id (str): Unique identifier for this election swarm
    """
    
    def __init__(
        self,
        agents: List[Agent],
        voting_method: str = "majority",  # majority, unanimous, quorum, ranked
        quorum_percentage: float = 0.51,
        max_voting_rounds: int = 3,
        tie_breaking_method: str = "random",  # random, weighted, judge
        judge_agent: Optional[Agent] = None,
        weights: Optional[List[float]] = None,
        output_type: str = "str",
        verbose: bool = False,
        print_on: bool = True,
        timeout: int = 60,  # seconds
    ):
        """
        Initialize an ElectionSwarm with voting agents and configuration.
        
        Args:
            agents: List of agent instances that will vote
            voting_method: Method to determine the winner
            quorum_percentage: Percentage of agents required for quorum (0.0 to 1.0)
            max_voting_rounds: Maximum number of voting rounds before fallback
            tie_breaking_method: Method to resolve ties
            judge_agent: Special agent to resolve ties when using "judge" tie-breaking
            weights: Optional weights for agents (must match agents list length)
            output_type: Format of the output
            verbose: Whether to print detailed logs
            print_on: Whether to print the results
            timeout: Maximum time in seconds to wait for all agents
        """
        if not agents or len(agents) < 2:
            raise ValueError("ElectionSwarm requires at least 2 agents")
            
        if weights and len(weights) != len(agents):
            raise ValueError("Weights list must match the number of agents")
            
        if tie_breaking_method == "judge" and not judge_agent:
            raise ValueError("Judge tie-breaking method requires a judge_agent")
            
        if quorum_percentage <= 0 or quorum_percentage > 1:
            raise ValueError("Quorum percentage must be between 0 and 1")
        
        # Initialize attributes
        self.agents = agents
        self.voting_method = voting_method
        self.quorum_percentage = quorum_percentage
        self.max_voting_rounds = max_voting_rounds
        self.tie_breaking_method = tie_breaking_method
        self.judge_agent = judge_agent
        self.weights = weights if weights else [1.0] * len(agents)
        self.output_type = output_type
        self.verbose = verbose
        self.print_on = print_on
        self.timeout = timeout
        
        # Track election state
        self.id = str(uuid.uuid4())
        self.election_history = []
        
        # Initialize conversation memory for the election process
        self.short_memory = Conversation(
            system_prompt=f"Election Swarm with {len(agents)} voting agents.",
            time_enabled=True,
            user="ElectionSwarm"
        )
        
    def run(
        self, 
        task: str,
        img: Optional[str] = None, 
        *args, 
        **kwargs
    ) -> Any:
        """
        Run the election process by having all agents vote on the task.
        
        Args:
            task: The task to be performed
            img: Optional image to be processed
            *args: Additional positional arguments to pass to agents
            **kwargs: Additional keyword arguments to pass to agents
            
        Returns:
            The election result based on the specified output_type
        """
        # Record election start
        election_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Log the election process
        if self.verbose:
            logger.info(f"Starting election {election_id} with task: {task}")
            
        self.short_memory.add(
            role="ElectionSwarm",
            content=f"Starting new election with task: {task}"
        )
        
        # Run all agents in parallel to get their responses
        responses = self._run_agents_concurrently(task, img, *args, **kwargs)
        
        # Process responses to determine the winner
        result = self._process_election(responses, task)
        
        # Record election end
        end_time = time.time()
        election_data = {
            "election_id": election_id,
            "task": task,
            "voting_method": self.voting_method,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "agent_responses": [
                {"agent_name": self.agents[i].agent_name, "response": response}
                for i, response in enumerate(responses) if response is not None
            ],
            "result": result,
        }
        
        self.election_history.append(election_data)
        
        # Add the result to memory
        self.short_memory.add(
            role="ElectionSwarm",
            content=f"Election result: {result}"
        )
        
        if self.print_on:
            formatter.print_panel(
                f"Election Complete\nTask: {task}\nResult: {result}",
                title=f"ElectionSwarm - {self.voting_method.title()} Voting"
            )
            
        return result
    
    def _run_agents_concurrently(
        self, 
        task: str, 
        img: Optional[str] = None,
        *args, 
        **kwargs
    ) -> List[Any]:
        """
        Run all agents concurrently and collect their responses.
        
        Uses ThreadPoolExecutor to run agents in parallel with a timeout.
        
        Args:
            task: The task to be performed by each agent
            img: Optional image input
            *args: Additional args to pass to agents
            **kwargs: Additional kwargs to pass to agents
            
        Returns:
            List of agent responses
        """
        executor = ThreadPoolExecutor(max_workers=len(self.agents))
        futures = []
        
        # Submit all agent tasks to the executor
        for agent in self.agents:
            if self.verbose:
                logger.info(f"Submitting task to agent: {agent.agent_name}")
                
            # Handle the case where agent might be a function rather than an Agent instance
            if callable(agent) and not isinstance(agent, Agent):
                future = executor.submit(agent, task, img, *args, **kwargs)
            else:
                future = executor.submit(agent.run, task, img, *args, **kwargs)
                
            futures.append(future)
        
        # Collect responses with timeout
        responses = []
        for future in futures:
            try:
                response = future.result(timeout=self.timeout)
                responses.append(response)
                
                if self.verbose:
                    logger.info(f"Agent response: {response[:100]}...")
                    
            except Exception as e:
                logger.error(f"Agent failed with error: {str(e)}")
                responses.append(None)
        
        executor.shutdown(wait=False)
        
        # Filter out None responses (failed agents)
        valid_responses = [r for r in responses if r is not None]
        
        if len(valid_responses) == 0:
            raise RuntimeError("All agents failed to respond")
            
        return responses
    
    def _process_election(self, responses: List[Any], original_task: str) -> Any:
        """
        Process agent responses according to the voting method to determine the result.
        
        Args:
            responses: List of agent responses
            original_task: The original task for reference in tie-breaking
            
        Returns:
            The election result
        """
        # Filter out None responses (failed agents)
        valid_responses = [r for r in responses if r is not None]
        
        if len(valid_responses) == 0:
            raise RuntimeError("All agents failed to respond")
            
        if self.voting_method == "majority":
            return self._majority_vote(valid_responses, original_task)
        elif self.voting_method == "unanimous":
            return self._unanimous_vote(valid_responses, original_task)
        elif self.voting_method == "quorum":
            return self._quorum_vote(valid_responses, original_task)
        elif self.voting_method == "ranked":
            return self._ranked_vote(valid_responses, original_task)
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")
    
    def _majority_vote(self, responses: List[Any], original_task: str) -> Any:
        """
        Determine the result by majority vote.
        
        Args:
            responses: List of valid agent responses
            original_task: Original task for reference in tie-breaking
            
        Returns:
            The winning response
        """
        # Count occurrences of each unique response
        vote_counts = {}
        
        # Convert responses to a consistent format for comparison
        normalized_responses = []
        for response in responses:
            normalized = self._normalize_response(response, original_task)
            normalized_responses.append(normalized)
            
            # Count votes
            if normalized in vote_counts:
                vote_counts[normalized] += 1
            else:
                vote_counts[normalized] = 1
        
        # Find the response with the most votes
        max_votes = max(vote_counts.values())
        top_responses = [resp for resp, count in vote_counts.items() if count == max_votes]
        
        # If there's a single winner, return it
        if len(top_responses) == 1:
            winning_idx = normalized_responses.index(top_responses[0])
            return responses[winning_idx]  # Return the original response
            
        # Otherwise, handle tie
        return self._break_tie(top_responses, original_task)
    
    def _unanimous_vote(self, responses: List[Any], original_task: str) -> Any:
        """
        Determine if there's a unanimous decision.
        
        Args:
            responses: List of valid agent responses
            original_task: Original task for reference if no unanimity
            
        Returns:
            The unanimous response or fallback to majority vote
        """
        # Check if all responses are identical
        normalized_responses = [self._normalize_response(r, original_task) for r in responses]
        
        # If all responses are identical, return the first response
        if len(set(normalized_responses)) == 1:
            return responses[0]
            
        # Otherwise fall back to majority vote
        self.short_memory.add(
            role="ElectionSwarm",
            content="No unanimous agreement, falling back to majority vote."
        )
        
        return self._majority_vote(responses, original_task)
    
    def _quorum_vote(self, responses: List[Any], original_task: str) -> Any:
        """
        Determine if any response meets the quorum percentage.
        
        Args:
            responses: List of valid agent responses
            original_task: Original task for reference if no quorum
            
        Returns:
            The response that meets quorum or fallback to majority vote
        """
        # Count occurrences of each unique response
        vote_counts = {}
        
        # Convert responses to a consistent format for comparison
        normalized_responses = []
        for response in responses:
            normalized = self._normalize_response(response, original_task)
            normalized_responses.append(normalized)
            
            # Count votes
            if normalized in vote_counts:
                vote_counts[normalized] += 1
            else:
                vote_counts[normalized] = 1
        
        # Check if any response meets the quorum percentage
        required_votes = len(responses) * self.quorum_percentage
        
        for resp, count in vote_counts.items():
            if count >= required_votes:
                # Return the original response
                winning_idx = normalized_responses.index(resp)
                return responses[winning_idx]
                
        # No quorum reached, fall back to majority vote
        self.short_memory.add(
            role="ElectionSwarm",
            content=f"No response reached quorum of {self.quorum_percentage*100}%, falling back to majority vote."
        )
        
        return self._majority_vote(responses, original_task)
    '''
    def _ranked_vote(self, responses: List[Any], original_task: str) -> Any:
        """
        Perform a ranked choice voting process.
        
        This requires additional voting rounds when no majority is found.
        
        Args:
            responses: List of valid agent responses
            original_task: Original task for all voting rounds
            
        Returns:
            The winning response after ranked choice voting
        """
        # Initial implementation simply defaults to majority voting for now
        # This could be expanded to a true ranked choice system in the future
        
        # For now, we'll implement a simple runoff system
        remaining_candidates = responses.copy()
        
        for round_num in range(self.max_voting_rounds):
            if len(remaining_candidates) <= 1:
                return remaining_candidates[0] if remaining_candidates else None
                
            # Count votes for remaining candidates
            vote_counts = {}
            for response in remaining_candidates:
                normalized = self._normalize_response(response, original_task)
                vote_counts[normalized] = vote_counts.get(normalized, 0) + 1
            
            # Find candidate with majority
            max_votes = max(vote_counts.values())
            if max_votes > len(remaining_candidates) / 2:
                # Majority found
                winning_response = [resp for resp, count in vote_counts.items() 
                                  if count == max_votes][0]
                                  
                # Return the original response
                for resp in responses:
                    if self._normalize_response(resp) == winning_response:
                        return resp
            
            # No majority, eliminate lowest-ranked candidate
            min_votes = min(vote_counts.values())
            to_eliminate = [resp for resp, count in vote_counts.items() 
                          if count == min_votes]
                          
            # Remove the eliminated candidates
            remaining_candidates = [
                resp for resp in remaining_candidates 
                if self._normalize_response(resp) not in to_eliminate
            ]
            
            self.short_memory.add(
                role="ElectionSwarm",
                content=f"Round {round_num+1}: Eliminated candidate(s) with {min_votes} votes."
            )
            
        # If we've reached max rounds without a winner, fall back to majority vote
        return self._majority_vote(responses, original_task)
    '''
    def _ranked_vote(self, responses: List[Any], original_task: str) -> Any:
        """
        执行排名选择投票过程，确保所有可能的执行路径都有有效返回值。
        """
        # 防御性编程 - 检查输入
        if not responses:
            self.short_memory.add(
                role="ElectionSwarm",
                content="No valid responses for ranked voting. Cannot proceed."
            )
            # 不应该返回 None - 明确返回一个错误消息或第一个响应
            return "No valid responses were provided for voting."
        
        # 初始化候选者
        remaining_candidates = responses.copy()
        
        # 主投票循环
        for round_num in range(self.max_voting_rounds):
            # 关键检查：如果候选者列表为空，提前返回
            if not remaining_candidates:
                self.short_memory.add(
                    role="ElectionSwarm",
                    content="All candidates eliminated with no clear winner. Falling back to majority vote."
                )
                # 回退到多数投票方法
                return self._majority_vote(responses, original_task)
                
            # 如果只剩一个候选者，返回该候选者
            if len(remaining_candidates) == 1:
                return remaining_candidates[0]
                
            # 计算剩余候选者的票数
            vote_counts = {}
            normalized_responses = []
            
            for response in remaining_candidates:
                normalized = self._normalize_response(response, original_task)
                normalized_responses.append(normalized)
                
                # 计票
                vote_counts[normalized] = vote_counts.get(normalized, 0) + 1
                
            # 如果投票结果为空（理论上不应该发生）
            if not vote_counts:
                self.short_memory.add(
                    role="ElectionSwarm",
                    content="No votes counted. Falling back to majority vote."
                )
                return self._majority_vote(responses, original_task)
            
            # 寻找多数派
            max_votes = max(vote_counts.values()) if vote_counts else 0
            if max_votes > len(remaining_candidates) / 2:
                # 找到多数派获胜者
                winning_responses = [resp for resp, count in vote_counts.items() if count == max_votes]
                
                if winning_responses:  # 应该总是True，但保险起见
                    winning_response = winning_responses[0]
                    
                    # 找到对应的原始响应
                    for resp in responses:
                        if self._normalize_response(resp, original_task) == winning_response:
                            return resp
                
                # 如果找不到对应的原始响应（几乎不可能），返回第一个剩余候选者
                return remaining_candidates[0]
            
            # 没有多数派，淘汰得票最少的候选者
            min_votes = min(vote_counts.values()) if vote_counts else 0
            to_eliminate = [resp for resp, count in vote_counts.items() if count == min_votes]
            
            # 重要改进：确保不会淘汰全部候选者
            # 如果所有候选者得票相同，只淘汰一半（避免全部淘汰）
            if len(to_eliminate) == len(vote_counts):
                import random
                to_eliminate = random.sample(to_eliminate, len(to_eliminate) // 2 + 1)
                
            # 移除被淘汰的候选者
            new_remaining = [resp for resp in remaining_candidates 
                            if self._normalize_response(resp, original_task) not in to_eliminate]
            
            # 检查是否有候选者被淘汰
            if len(new_remaining) == len(remaining_candidates):
                # 没有候选者被淘汰（可能逻辑错误），为避免死循环，强制淘汰一个
                if remaining_candidates:
                    import random
                    new_remaining.remove(random.choice(remaining_candidates))
                    
            remaining_candidates = new_remaining
            
            # 记录每轮淘汰情况
            self.short_memory.add(
                role="ElectionSwarm",
                content=f"Round {round_num+1}: Eliminated candidate(s) with {min_votes} votes. Remaining: {len(remaining_candidates)}"
            )
        
        # 重要：循环结束后必须有明确的返回语句
        # 如果达到最大轮数仍无结果，回退到多数投票
        self.short_memory.add(
            role="ElectionSwarm",
            content=f"Reached maximum {self.max_voting_rounds} voting rounds without a winner. Falling back to majority vote."
        )
        return self._majority_vote(responses, original_task)

    def _break_tie(self, tied_responses: List[Any], original_task: str) -> Any:
        """
        Break a tie according to the configured tie-breaking method.
        
        Args:
            tied_responses: List of responses that are tied
            original_task: Original task for reference
            
        Returns:
            The winning response after tie-breaking
        """
        if self.tie_breaking_method == "random":
            import random
            return random.choice(tied_responses)
            
        elif self.tie_breaking_method == "weighted":
            # Use agent weights to break tie
            max_weight = 0
            winner = None
            
            for i, response in enumerate(tied_responses):
                weight = self.weights[i]
                if weight > max_weight:
                    max_weight = weight
                    winner = response
            
            return winner
            
        elif self.tie_breaking_method == "judge":
            # Use judge agent to decide
            if not self.judge_agent:
                raise ValueError("Judge tie-breaking requires a judge agent")
                
            judge_task = f"""
            You are serving as a judge to break a tie in an election.
            
            The original task was: "{original_task}"
            
            The following responses tied for first place:
            
            {self._format_tied_responses(tied_responses)}
            
            Which response best addresses the original task? Respond only with the number of your choice.
            """
            
            judge_decision = self.judge_agent.run(judge_task)
            
            # Extract numeric decision from judge response
            try:
                decision_num = int(''.join(filter(str.isdigit, judge_decision)))
                if 1 <= decision_num <= len(tied_responses):
                    return tied_responses[decision_num - 1]
            except:
                # If judge response couldn't be interpreted, fall back to random
                import random
                return random.choice(tied_responses)
        
        # Default fallback
        import random
        return random.choice(tied_responses)
    
    def _format_tied_responses(self, tied_responses: List[Any]) -> str:
        """Format tied responses for the judge agent."""
        formatted = ""
        for i, response in enumerate(tied_responses, 1):
            formatted += f"Response {i}:\n{response}\n\n"
        return formatted
    '''
    def _normalize_response(self, response: Any) -> str:
        """
        Normalize responses to ensure consistent comparison.
        
        Args:
            response: The response to normalize
            
        Returns:
            Normalized string representation of the response
        """
        # Convert response to string and normalize whitespace
        resp_str = any_to_str(response).strip()
        
        # Remove extra whitespace and convert to lowercase for comparison
        normalized = " ".join(resp_str.lower().split())
        
        return normalized
    '''
    def _normalize_response(self, response: Any, task: str = "") -> str:
        """
        Normalize responses to ensure consistent comparison and remove irrelevant content.
        
        Args:
            response: The response to normalize
            task: The current task to focus on
            
        Returns:
            Normalized string representation of the response relevant to the task
        """
        # Convert response to string and normalize whitespace
        resp_str = any_to_str(response).strip()
        
        # Remove extra whitespace and convert to lowercase for comparison
        normalized = " ".join(resp_str.lower().split())
        
        # Try to filter out content not relevant to the current task
        if task:
            # Attempt to extract only the relevant part of the response
            task_lower = task.lower()
            
            # Check if the task appears in the response
            if task_lower in normalized:
                # Keep only what comes after the task mention
                parts = normalized.split(task_lower, 1)
                if len(parts) > 1:
                    normalized = parts[1].strip()
            
            # Check for common patterns indicating answers/responses
            response_indicators = ["answer:", "response:", "result:", "# ", "## "]
            for indicator in response_indicators:
                if indicator in normalized:
                    # Find the last occurrence of the indicator
                    parts = normalized.split(indicator)
                    if len(parts) > 1:
                        # Keep only the last occurrence and everything after it
                        normalized = indicator + parts[-1]
                        break
        return normalized
        
    async def arun(
        self, 
        task: str,
        img: Optional[str] = None, 
        *args, 
        **kwargs
    ) -> Any:
        """
        Asynchronous version of run method.
        
        Args:
            task: The task to be performed
            img: Optional image to be processed
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            The election result
        """
        return await asyncio.to_thread(
            self.run,
            task=task,
            img=img,
            *args,
            **kwargs
        )
        
    def add_agent(self, agent: Agent, weight: float = 1.0):
        """
        Add a new agent to the election swarm.
        
        Args:
            agent: The agent to add
            weight: The voting weight for this agent
        """
        self.agents.append(agent)
        self.weights.append(weight)
        
    def remove_agent(self, agent: Agent):
        """
        Remove an agent from the election swarm.
        
        Args:
            agent: The agent to remove
        """
        if agent in self.agents:
            idx = self.agents.index(agent)
            self.agents.pop(idx)
            self.weights.pop(idx)
        else:
            raise ValueError("Agent not found in election swarm")
            
    def get_election_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about past elections.
        
        Returns:
            Dictionary with election statistics
        """
        if not self.election_history:
            return {"total_elections": 0}
            
        total_elections = len(self.election_history)
        total_duration = sum(e["duration"] for e in self.election_history)
        avg_duration = total_duration / total_elections
        
        return {
            "total_elections": total_elections,
            "total_duration": total_duration,
            "average_duration": avg_duration,
            "voting_method": self.voting_method,
        }
        
    def get_last_election_details(self) -> Dict[str, Any]:
        """
        Get details of the last election.
        
        Returns:
            Dictionary with details of the last election
        """
        if not self.election_history:
            return {"status": "No elections held"}
            
        return self.election_history[-1]
        
    def __call__(
        self, 
        task: str,
        img: Optional[str] = None, 
        *args, 
        **kwargs
    ) -> Any:
        """Make the ElectionSwarm callable like a function."""
        return self.run(task, img, *args, **kwargs)
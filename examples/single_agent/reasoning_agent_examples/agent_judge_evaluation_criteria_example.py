"""
Agent Judge with Evaluation Criteria Example

This example demonstrates how to use the AgentJudge with custom evaluation criteria.
The evaluation_criteria parameter allows specifying different criteria with weights 
for more targeted and customizable evaluation of agent outputs.
"""

from swarms.agents.agent_judge import AgentJudge
import os
from dotenv import load_dotenv

load_dotenv()

# Example 1: Basic usage with evaluation criteria
print("\n=== Example 1: Using Custom Evaluation Criteria ===\n")

# Create an AgentJudge with custom evaluation criteria
judge = AgentJudge(
    model_name="claude-3-7-sonnet-20250219",  # Use any available model
    evaluation_criteria={
        "correctness": 0.5,
        "problem_solving_approach": 0.3, 
        "explanation_clarity": 0.2
    }
)

# Sample output to evaluate
task_response = [
    "Task: Determine the time complexity of a binary search algorithm and explain your reasoning.\n\n"
    "Agent response: The time complexity of binary search is O(log n). In each step, "
    "we divide the search space in half, resulting in a logarithmic relationship between "
    "the input size and the number of operations. This can be proven by solving the "
    "recurrence relation T(n) = T(n/2) + O(1), which gives us T(n) = O(log n)."
]

# Run evaluation
evaluation = judge.run(task_response)
print(evaluation[0])

# Example 2: Specialized criteria for code evaluation
print("\n=== Example 2: Code Evaluation with Specialized Criteria ===\n")

code_judge = AgentJudge(
    model_name="claude-3-7-sonnet-20250219",
    agent_name="code_judge",
    evaluation_criteria={
        "code_correctness": 0.4,
        "code_efficiency": 0.3,
        "code_readability": 0.3
    }
)

# Sample code to evaluate
code_response = [
    "Task: Write a function to find the maximum subarray sum in an array of integers.\n\n"
    "Agent response:\n```python\n"
    "def max_subarray_sum(arr):\n"
    "    current_sum = max_sum = arr[0]\n"
    "    for i in range(1, len(arr)):\n"
    "        current_sum = max(arr[i], current_sum + arr[i])\n"
    "        max_sum = max(max_sum, current_sum)\n"
    "    return max_sum\n\n"
    "# Example usage\n"
    "print(max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]))  # Output: 6 (subarray [4, -1, 2, 1])\n"
    "```\n"
    "This implementation uses Kadane's algorithm which has O(n) time complexity and "
    "O(1) space complexity, making it optimal for this problem."
]

code_evaluation = code_judge.run(code_response)
print(code_evaluation[0])

# Example 3: Comparing multiple responses
print("\n=== Example 3: Comparing Multiple Agent Responses ===\n")

comparison_judge = AgentJudge(
    model_name="claude-3-7-sonnet-20250219",
    evaluation_criteria={
        "accuracy": 0.6,
        "completeness": 0.4
    }
)

multiple_responses = comparison_judge.run([
    "Task: Explain the CAP theorem in distributed systems.\n\n"
    "Agent A response: CAP theorem states that a distributed system cannot simultaneously "
    "provide Consistency, Availability, and Partition tolerance. In practice, you must "
    "choose two out of these three properties.",
    
    "Task: Explain the CAP theorem in distributed systems.\n\n"
    "Agent B response: The CAP theorem, formulated by Eric Brewer, states that in a "
    "distributed data store, you can only guarantee two of the following three properties: "
    "Consistency (all nodes see the same data at the same time), Availability (every request "
    "receives a response), and Partition tolerance (the system continues to operate despite "
    "network failures). Most modern distributed systems choose to sacrifice consistency in "
    "favor of availability and partition tolerance, implementing eventual consistency models instead."
])

print(multiple_responses[0])
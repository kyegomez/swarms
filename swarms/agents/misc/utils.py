from abc import ABC, abstractmethod


class LeaderAgent(ABC):
    @abstractmethod
    def distribute_task(self, WAs, task):
        pass

    @abstractmethod
    def collect_results(self, WAs):
        pass

    @abstractmethod
    def process_results(self):
        pass


class WorkerAgent(ABC):
    @abstractmethod
    def execute_task(self):
        pass


class CollaborativeAgent(ABC):
    @abstractmethod
    def execute_task(self, task):
        pass

    @abstractmethod
    def collaborate(self):
        pass


class CompetitiveAgent(ABC):
    @abstractmethod
    def execute_task(self, task):
        pass


def evaluate_results(CompAs):
    pass



# Example
class MyWorkerAgent(WorkerAgent):
    def execute_task(self):
        # Insert your implementation here
        pass



import json
import logging
from typing import Any, Dict, List

from tiktoken import Tokenizer, TokenizerException


# Helper function to count tokens
def count_tokens(text: str, tokenizer: Tokenizer) -> int:
    try:
        tokens = tokenizer.tokenize(text)
        return len(tokens)
    except TokenizerException as e:
        logging.error(f"Error tokenizing text: {e}")
        return 0

def divide_and_conquer_v2(task: str, agents_memory: List[Dict[str, Any]], max_tokens: int = 8192) -> str:
    """
    This function divides a complex task into smaller subtasks and assigns each subtask to a different agent.
    Then, it combines the results to form the final solution, considering the GPT-4 token limit.

    Args:
        task (str): The complex task to be solved.
        agents_memory (List[Dict[str, Any]]): A list of agent memory states.
        max_tokens (int, optional): The maximum number of tokens GPT-4 can handle. Defaults to 8192.

    Returns:
        str: The final solution to the complex task.

    Error handling:
        If the text exceeds the token limit, an error message is logged, and the function returns an empty string.
    """

    logging.info(f"Divide and conquer started for task: {task}")

    subtasks = split_task_into_subtasks(task)
    results = []

    tokenizer = Tokenizer()

    for subtask in subtasks:
        agent_memory = random.choice(agents_memory)
        chat_input = agent_memory + [{"role": "user", "content": subtask}]
        tokens = count_tokens(json.dumps(chat_input), tokenizer)

        if tokens >= max_tokens:
            logging.error("Token limit exceeded for divide_and_conquer_v2")
            return ""

        result, _ = chat(chat_input)
        results.append(result.strip())

    final_solution = combine_subtask_results(results)
    logging.info(f"Divide and conquer completed. Final solution: {final_solution}")

    # Save the final solution to a database (e.g., a document-based database like MongoDB)
    save_solution_to_database(task, final_solution)

    return final_solution

def collaborative_execution_v2(task: str, agents_memory: List[Dict[str, Any]], max_tokens: int = 8192) -> str:
    """
    This function allows a group of agents to collaborate on solving a complex task, considering the GPT-4 token limit.
    Each agent proposes a solution, and a final solution is derived from the best aspects of each proposal.

    Args:
        task (str): The complex task to be solved.
        agents_memory (List[Dict[str, Any]]): A list of agent memory states.
        max_tokens (int, optional): The maximum number of tokens GPT-4 can handle. Defaults to 8192.

    Returns:
        str: The final solution to the complex task.

    Error handling:
        If the text exceeds the token limit, an error message is logged, and the function returns an empty string.
    """

    logging.info(f"Collaborative execution started for task: {task}")

    solutions = []
    tokenizer = Tokenizer()

    for agent_memory in agents_memory:
        chat_input = agent_memory + [{"role": "user", "content": f"Propose a solution for: {task}"}]
        tokens = count_tokens(json.dumps(chat_input), tokenizer)

        if tokens >= max_tokens:
            logging.error("Token limit exceeded for collaborative_execution_v2")
            return ""

        solution, _ = chat(chat_input)
        solutions.append({"role": "assistant", "content": solution.strip()})

    chat_input = [{"role": "system", "content": "You are a collaborative AI agent. Work with other agents to solve the given task."}] + solutions + [{"role": "user", "content": "Combine the best aspects of each solution to create the final solution."}]
    tokens = count_tokens(json.dumps(chat_input), tokenizer)

    if tokens >= max_tokens:
        logging.error("Token limit exceeded for collaborative_execution_v2")
        return ""

    final_solution, _ = chat(chat_input)

    logging.info(f"Collaborative execution completed. Final solution: {final_solution}")

    # Save the final solution to a database (e.g., a graph-based database like Neo4j for better analysis of connections)
    save_solution_to_database(task, final_solution)

    return final_solution.strip()


def expert_agents_v2(task: str, domain_experts_memory: List[Dict[str, Any]], max_tokens: int = 8192) -> str:
    """
    This function allows a group of domain expert agents to provide solutions to a given task.
    The function evaluates the quality of each solution and returns the best one.

    Args:
        task (str): The complex task to be solved.
        domain_experts_memory (List[Dict[str, Any]]): A list of domain expert agent memory states.
        max_tokens (int, optional): The maximum number of tokens GPT-4 can handle. Defaults to 8192.

    Returns:
        str: The best solution to the complex task.

    Error handling:
        If the text exceeds the token limit, an error message is logged, and the function returns an empty string.
    """

    logging.info(f"Expert agents execution started for task: {task}")

    best_solution = None
    best_score = 0
    tokenizer = Tokenizer()

    for expert_memory in domain_experts_memory:
        chat_input = expert_memory + [{"role": "user", "content": f"Provide a solution for: {task} based on your domain expertise."}]
        tokens = count_tokens(json.dumps(chat_input), tokenizer)

        if tokens >= max_tokens:
            logging.error("Token limit exceeded for expert_agents_v2")
            return ""

        expert_solution, _ = chat(chat_input)
        score = evaluate_solution_quality(task, expert_solution.strip())

        if score > best_score:
            best_solution = expert_solution.strip()
            best_score = score

    logging.info(f"Expert agents execution completed. Best solution: {best_solution}")

    # Save the best solution to a database (e.g., a relational database like PostgreSQL for structured data)
    save_solution_to_database(task, best_solution)

    return best_solution


def _v2(taskagent_delegation: str, manager_agents_memory: List[Dict[str, Any]], subordinate_agents_memory: List[Dict[str, Any]], max_tokens: int = 8192) -> str:
    """
    This function allows a group of manager agents to delegate a complex task to a group of subordinate agents.
    Each manager agent selects the best subordinate agent for each subtask, and the results are combined.

    Args:
        task (str): The complex task to be solved.
        manager_agents_memory (List[Dict[str, Any]]): A list of manager agent memory states.
        subordinate_agents_memory (List[Dict[str, Any]]): A list of subordinate agent memory states.
        max_tokens (int, optional): The maximum number of tokens GPT-4 can handle. Defaults to 8192.

    Returns:
        str: The final combined result of the complex task.

    Error handling:
        If the text exceeds the token limit, an error message is logged, and the function returns an empty string.
    """

    logging.info(f"Agent delegation execution started for task: {task}")

    subtasks = generate_tasks(task)
    results = []

    for subtask in subtasks:
        manager_memory = random.choice(manager_agents_memory)
        selected_subordinate_memory = None

        while selected_subordinate_memory is None:
            chat_input = manager_memory + [{"role": "user", "content": f"Select the best subordinate to solve: {subtask}"}]
            tokens = count_tokens(json.dumps(chat_input), tokenizer)

            if tokens >= max_tokens:
                logging.error("Token limit exceeded for agent_delegation_v2")
                return ""

            suggested_subordinate, _ = chat(chat_input)
            subordinate_id = int(suggested_subordinate.strip())

            if 0 <= subordinate_id < len(subordinate_agents_memory):
                selected_subordinate_memory = subordinate_agents_memory[subordinate_id]
            else:
                manager_memory.append({"role": "assistant", "content": "Invalid subordinate ID, please try again."})

        result = continue_until_done(subtask, selected_subordinate_memory)
        results.append(result)

    final_result = combine_results(results)

    logging.info(f"Agent delegation execution completed. Final result: {final_result}")

    # Save the final result to a database (e.g., a graph database like Neo4j for mapping connections between entities)
    save_result_to_database(task, final_result)

    return final_result


def parallel_execution_v2(task: str, num_agents: int, agents_memory: List[Dict[str, Any]], max_tokens: int = 8192) -> str:
    """
    This function uses multiple agents to solve a complex task in parallel.
    Each agent works on a subtask, and the results are combined.

    Args:
        task (str): The complex task to be solved.
        num_agents (int): The number of agents working in parallel.
        agents_memory (List[Dict[str, Any]]): A list of agent memory states.
        max_tokens (int, optional): The maximum number of tokens GPT-4 can handle. Defaults to 8192.

    Returns:
        str: The final combined result of the complex task.

    Error handling:
        If the text exceeds the token limit, an error message is logged, and the function returns an empty string.
    """

    logging.info(f"Parallel execution started for task: {task}")

    tasks = generate_tasks(task)
    results = []
    threads = []

    def threaded_execution(task: str, agent_memory: Dict[str, Any], results: List[str]) -> None:
        chat_input = agent_memory + [{"role": "user", "content": task}]
        tokens = count_tokens(json.dumps(chat_input), tokenizer)

        if tokens >= max_tokens:
            logging.error("Token limit exceeded for parallel_execution_v2")
            return

        result = continue_until_done(task, agent_memory)
        results.append(result)

    for task in tasks:
        agent_id = random.randint(0, num_agents - 1)
        t = threading.Thread(target=threaded_execution, args=(task, agents_memory[agent_id], results))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    final_result = combine_results(results)

    logging.info(f"Parallel execution completed. Final result: {final_result}")

    # Save the final result to a database (e.g., a relational database like PostgreSQL for structured data)
    save_result_to_database(task, final_result)

    return final_result


def hierarchical_execution_v2(task: str, num_levels: int, agents_memory: List[Dict[str, Any]], max_tokens: int = 8192) -> str:
    """
    This function solves a complex task by dividing it into smaller subtasks and assigning them to agents in a
    hierarchical manner.

    Args:
        task (str): The complex task to be solved.
        num_levels (int): The number of hierarchical levels in the agent hierarchy.
        agents_memory (List[Dict[str, Any]]): A list of agent memory states.
        max_tokens (int, optional): The maximum number of tokens GPT-4 can handle. Defaults to 8192.

    Returns:
        str: The final combined result of the complex task.

    Error handling:
        If the text exceeds the token limit, an error message is logged, and the function returns an empty string.
    """

    logging.info(f"Hierarchical execution started for task: {task}")

    levels = divide_problem_into_modules(task)
    results = []

    for level in levels:
        assigned_agent_memory = agents_memory[num_levels % len(agents_memory)]
        chat_input = assigned_agent_memory + [{"role": "user", "content": level}]
        tokens = count_tokens(json.dumps(chat_input), tokenizer)

        if tokens >= max_tokens:
            logging.error("Token limit exceeded for hierarchical_execution_v2")
            return ""

        result = continue_until_done(level, assigned_agent_memory)
        results.append(result)
        num_levels += 1

    final_result = combine_results(results)

    logging.info(f"Hierarchical execution completed. Final result: {final_result}")

    # Save the final result to a database (e.g., a graph database like Neo4j for hierarchical relationships)
    save_result_to_database(task, final_result)

    return final_result


def consensus_based_decision_v2(task_prompt: str, agents_memory: List[Dict[str, Any]], max_tokens: int = 8192) -> str:
    """
    This function takes a task prompt and a list of agent memories, and it returns the consensus-based decision among
    the agents.

    Args:
        task_prompt (str): The task prompt to be solved.
        agents_memory (List[Dict[str, Any]]): A list of agent memory states.
        max_tokens (int, optional): The maximum number of tokens GPT-4 can handle. Defaults to 8192.

    Returns:
        str: The consensus-based decision among the agents.

    Error handling:
        If the text exceeds the token limit, an error message is logged, and the function returns an empty string.
    """

    logging.info(f"Consensus-based decision started for task: {task_prompt}")

    options = collaborative_brainstorm(task_prompt, agents_memory[0], agents_memory[1])
    votes = []

    for option in options:
        vote_count = 0

        for agent_memory in agents_memory:
            chat_input = agent_memory + [{"role": "user", "content": f"Which option do you prefer: {options[0]} or {option}?"}]
            tokens = count_tokens(json.dumps(chat_input), tokenizer)

            if tokens >= max_tokens:
                logging.error("Token limit exceeded for consensus_based_decision_v2")
                return ""

            vote, _ = chat(chat_input)
            if vote.strip() == option:
                vote_count += 1

        votes.append(vote_count)

    consensus_option = options[votes.index(max(votes))]

    logging.info(f"Consensus-based decision completed. Final result: {consensus_option}")

    # Save the final result to a database (e.g., a relational database like PostgreSQL for structured data)
    save_result_to_database(task_prompt, consensus_option)

    return consensus_option


def ask_for_help_v2(chatbot1_memory: List[Dict[str, Any]], chatbot2_memory: List[Dict[str, Any]], max_tokens: int = 8192) -> str:
    """
    This function facilitates the interaction between two chatbots. Chatbot1 asks Chatbot2 for help on a task.

    Args:
        chatbot1_memory (List[Dict[str, Any]]): Memory state of Chatbot1.
        chatbot2_memory (List[Dict[str, Any]]): Memory state of Chatbot2.
        max_tokens (int, optional): The maximum number of tokens GPT-4 can handle. Defaults to 8192.

    Returns:
        str: The suggestion provided by Chatbot2 to help Chatbot1 with the task.

    Error handling:
        If the text exceeds the token limit, an error message is logged, and the function returns an empty string.
    """

    logging.info("Ask for help started")

    chat_input1 = chatbot1_memory + [{"role": "user", "content": "Chatbot1, I need help with this task."}]
    tokens1 = count_tokens(json.dumps(chat_input1), tokenizer)

    if tokens1 >= max_tokens:
        logging.error("Token limit exceeded for ask_for_help_v2")
        return ""

    chatbot1_help_request, chatbot1_tokens = chat(chat_input1)
    chatbot1_memory.append({"role": "assistant", "content": chatbot1_help_request})
    
    chat_input2 = chatbot2_memory + [{"role": "user", "content": f"Chatbot2, please help me with this: {chatbot1_help_request}"}]
    tokens2 = count_tokens(json.dumps(chat_input2), tokenizer)

    if tokens2 >= max_tokens:
        logging.error("Token limit exceeded for ask_for_help_v2")
        return ""

    chatbot2_suggestion, chatbot2_tokens = chat(chat_input2)
    chatbot2_memory.append({"role": "assistant", "content": chatbot2_suggestion})

    logging.info(f"Ask for help completed. Chatbot2's suggestion: {chatbot2_suggestion}")

    # Save the chat history to a database (e.g., a graph database like Neo4j for interconnected data)
    save_chat_history_to_database(chatbot1_memory, chatbot2_memory)

    return chatbot2_suggestion


def collaborative_brainstorm_v2(topic: str, chatbot1_memory: List[Dict[str, Any]], chatbot2_memory: List[Dict[str, Any]], max_tokens: int = 8192) -> List[str]:
    """
    This function enables two chatbots to collaboratively brainstorm ideas on a given topic.

    Args:
        topic (str): The topic for brainstorming.
        chatbot1_memory (List[Dict[str, Any]]): Memory state of Chatbot1.
        chatbot2_memory (List[Dict[str, Any]]): Memory state of Chatbot2.
        max_tokens (int, optional): The maximum number of tokens GPT-4 can handle. Defaults to 8192.

    Returns:
        List[str]: A list of brainstormed ideas from both chatbots.

    Error handling:
        If the text exceeds the token limit, an error message is logged, and the function returns an empty list.
    """

    logging.info(f"Collaborative brainstorming started for topic: {topic}")

    ideas = []

    for i in range(3):
        chat_input1 = chatbot1_memory + [{"role": "user", "content": f"Chatbot1, brainstorm an idea for {topic}"}]
        tokens1 = count_tokens(json.dumps(chat_input1), tokenizer)

        if tokens1 >= max_tokens:
            logging.error("Token limit exceeded for collaborative_brainstorm_v2")
            return []

        chatbot1_idea, chatbot1_tokens = chat(chat_input1)
        chatbot1_memory.append({"role": "assistant", "content": chatbot1_idea})
        ideas.append(chatbot1_idea)

        chat_input2 = chatbot2_memory + [{"role": "user", "content": f"Chatbot2, brainstorm an idea for {topic}"}]
        tokens2 = count_tokens(json.dumps(chat_input2), tokenizer)

        if tokens2 >= max_tokens:
            logging.error("Token limit exceeded for collaborative_brainstorm_v2")
            return []

        chatbot2_idea, chatbot2_tokens = chat(chat_input2)
        chatbot2_memory.append({"role": "assistant", "content": chatbot2_idea})
        ideas.append(chatbot2_idea)

    logging.info(f"Collaborative brainstorming completed. Ideas: {ideas}")

    # Save the brainstorming session to a database (e.g., a document database like MongoDB for storing complex data structures)
    save_brainstorming_session_to_database(topic, ideas, chatbot1_memory, chatbot2_memory)

    return ideas


def graph_based_chat_v2(chatbot_memory: List[Dict[str, Any]], user_id: str, user_message: str, graph_database: GraphDatabase, max_tokens: int = 8192) -> str:
    """
    This function allows a chatbot to engage in a conversation with a user, utilizing a graph database to provide insights
    and connections between users, keywords, and topics.

    Args:
        chatbot_memory (List[Dict[str, Any]]): Memory state of the chatbot.
        user_id (str): The unique identifier for the user.
        user_message (str): The message from the user.
        graph_database (GraphDatabase): The graph database containing connections between users, keywords, topics, etc.
        max_tokens (int, optional): The maximum number of tokens GPT-4 can handle. Defaults to 8192.

    Returns:
        str: The chatbot's response to the user's message.

    Error handling:
        If the text exceeds the token limit, an error message is logged, and the function returns an empty string.
    """

    logging.info(f"Received message from user {user_id}: {user_message}")

    # Update the graph database with user's message
    update_graph_database(user_id, user_message, graph_database)

    # Retrieve insights from the graph database
    insights = get_insights(graph_database)

    chat_input = chatbot_memory + [{"role": "user", "content": f"{user_message}\nInsights: {insights}"}]
    tokens = count_tokens(json.dumps(chat_input), tokenizer)

    if tokens >= max_tokens:
        logging.error("Token limit exceeded for graph_based_chat_v2")
        return ""

    chatbot_response, chatbot_tokens = chat(chat_input)
    chatbot_memory.append({"role": "assistant", "content": chatbot_response})

    logging.info(f"Chatbot response to user {user_id}: {chatbot_response}")

    # Save the chat conversation to a database (e.g., a relational database like MySQL for structured data)
    save_chat_conversation_to_database(user_id, user_message, chatbot_response)

    return chatbot_response


def multi_platform_chat_v2(platform: str, chatbot_memory: List[Dict[str, Any]], user_id: str, user_message: str, max_tokens: int = 8192) -> str:
    """
    This function allows a chatbot to engage in a conversation with a user on various platforms such as
    WhatsApp, Snapchat, Facebook, Twitter, etc.

    Args:
        platform (str): The platform on which the chat is taking place (e.g., "WhatsApp", "Facebook").
        chatbot_memory (List[Dict[str, Any]]): Memory state of the chatbot.
        user_id (str): The unique identifier for the user.
        user_message (str): The message from the user.
        max_tokens (int, optional): The maximum number of tokens GPT-4 can handle. Defaults to 8192.

    Returns:
        str: The chatbot's response to the user's message.

    Error handling:
        If the text exceeds the token limit, an error message is logged, and the function returns an empty string.
    """

    logging.info(f"Received message from user {user_id} on {platform}: {user_message}")

    chat_input = chatbot_memory + [{"role": "user", "content": f"Platform: {platform}\nUser message: {user_message}"}]
    tokens = count_tokens(json.dumps(chat_input), tokenizer)

    if tokens >= max_tokens:
        logging.error("Token limit exceeded for multi_platform_chat_v2")
        return ""

    chatbot_response, chatbot_tokens = chat(chat_input)
    chatbot_memory.append({"role": "assistant", "content": chatbot_response})

    logging.info(f"Chatbot response to user {user_id} on {platform}: {chatbot_response}")

    # Save the chat conversation to a database (e.g., a document-based database like MongoDB for unstructured data)
    save_chat_conversation_to_database(user_id, platform, user_message, chatbot_response)

    return chatbot_response


def agent_swapping_v2(task_prompt: str, agents_memory: List[Dict[str, Any]], max_tokens: int = 8192) -> str:
    """
    This function allows multiple agents to collaboratively solve a task by swapping in and out when
    their individual knowledge is insufficient.

    Args:
        task_prompt (str): The task to be solved.
        agents_memory (List[Dict[str, Any]]): List of memory states for each agent.
        max_tokens (int, optional): The maximum number of tokens GPT-4 can handle. Defaults to 8192.

    Returns:
        str: The final solution to the task.

    Error handling:
        If the text exceeds the token limit, an error message is logged, and the function returns an empty string.
    """

    current_agent_index = 0
    current_agent_memory = agents_memory[current_agent_index]
    input_messages = current_agent_memory + [{"role": "user", "content": f"Task: {task_prompt}"}]
    tokens = count_tokens(json.dumps(input_messages), tokenizer)

    if tokens >= max_tokens:
        logging.error("Token limit exceeded for agent_swapping_v2")
        return ""

    partial_solution, remaining_task = chat(input_messages)

    while remaining_task:
        current_agent_index = (current_agent_index + 1) % len(agents_memory)
        current_agent_memory = agents_memory[current_agent_index]
        input_messages = current_agent_memory + [{"role": "user", "content": f"Remaining task: {remaining_task}"}]
        tokens = count_tokens(json.dumps(input_messages), tokenizer)

        if tokens >= max_tokens:
            logging.error("Token limit exceeded for agent_swapping_v2")
            return ""

        next_partial_solution, remaining_task = chat(input_messages)
        partial_solution += next_partial_solution

    return partial_solution


def multi_agent_voting_v2(task_prompt: str, agents_memory: List[Dict[str, Any]], max_tokens: int = 8192) -> str:
    """
    This function allows multiple agents to collaboratively solve a task by proposing solutions and
    voting on the best one.

    Args:
        task_prompt (str): The task to be solved.
        agents_memory (List[Dict[str, Any]]): List of memory states for each agent.
        max_tokens (int, optional): The maximum number of tokens GPT-4 can handle. Defaults to 8192.

    Returns:
        str: The final solution to the task.

    Error handling:
        If the text exceeds the token limit, an error message is logged, and the function returns an empty string.
    """

    proposed_solutions = []
    for agent_memory in agents_memory:
        input_messages = agent_memory + [{"role": "user", "content": f"Propose a solution for: {task_prompt}"}]
        tokens = count_tokens(json.dumps(input_messages), tokenizer)

        if tokens >= max_tokens:
            logging.error("Token limit exceeded for multi_agent_voting_v2")
            return ""

        proposed_solution, _ = chat(input_messages)
        proposed_solutions.append(proposed_solution.strip())

    input_messages = [{"role": "system", "content": "You are an AI agent. Vote on the best solution from the following options:"}] + [{"role": "assistant", "content": option} for option in proposed_solutions]
    tokens = count_tokens(json.dumps(input_messages), tokenizer)

    if tokens >= max_tokens:
        logging.error("Token limit exceeded for multi_agent_voting_v2")
        return ""

    winning_solution, _ = chat(input_messages + [{"role": "user", "content": "Which solution is the best?"}])
    return winning_solution.strip()


def multi_agent_brainstorming_v2(topic: str, agents_memory: List[Dict[str, Any]], max_tokens: int = 8192) -> List[str]:
    """
    This function allows multiple agents to collaboratively brainstorm ideas on a given topic.

    Args:
        topic (str): The topic for brainstorming.
        agents_memory (List[Dict[str, Any]]): List of memory states for each agent.
        max_tokens (int, optional): The maximum number of tokens GPT-4 can handle. Defaults to 8192.

    Returns:
        List[str]: List of brainstormed ideas.

    Error handling:
        If the text exceeds the token limit, an error message is logged, and the function returns an empty list.
    """

    ideas = []

    for agent_memory in agents_memory:
        input_messages = agent_memory + [{"role": "user", "content": f"Brainstorm an idea for: {topic}"}]
        tokens = count_tokens(json.dumps(input_messages), tokenizer)

        if tokens >= max_tokens:
            logging.error("Token limit exceeded for multi_agent_brainstorming_v2")
            return []

        idea, _ = chat(input_messages)
        ideas.append(idea.strip())

    return ideas


def multi_agent_emotion_analysis_v2(text: str, agents_memory: List[Dict[str, Any]], max_tokens: int = 8192) -> Dict[str, float]:
    """
    This function allows multiple agents to perform emotion analysis on a given text.

    Args:
        text (str): The text to perform emotion analysis on.
        agents_memory (List[Dict[str, Any]]): List of memory states for each agent.
        max_tokens (int, optional): The maximum number of tokens GPT-4 can handle. Defaults to 8192.

    Returns:
        Dict[str, float]: A dictionary containing emotion scores for the text.

    Error handling:
        If the text exceeds the token limit, an error message is logged, and the function returns an empty dictionary.
    """

    emotion_scores = defaultdict(float)

    for agent_memory in agents_memory:
        input_messages = agent_memory + [{"role": "user", "content": f"Analyze the emotions in this text: {text}"}]
        tokens = count_tokens(json.dumps(input_messages), tokenizer)

        if tokens >= max_tokens:
            logging.error("Token limit exceeded for multi_agent_emotion_analysis_v2")
            return {}

        emotion_analysis, _ = chat(input_messages)
        parsed_scores = json.loads(emotion_analysis.strip())

        for emotion, score in parsed_scores.items():
            emotion_scores[emotion] += score

    for emotion in emotion_scores:
        emotion_scores[emotion] /= len(agents_memory)

    return emotion_scores

def swarm_intelligence(task_prompt, agents_memory):
    subtasks = generate_tasks(task_prompt)
    results = []
    for subtask in subtasks:
        agent_votes = []
        for agent_memory in agents_memory:
            agent_vote, _ = chat(agent_memory + [{"role": "user", "content": f"Propose a solution for: {subtask}"}])
            agent_votes.append(agent_vote.strip())
        most_common_solution = max(set(agent_votes), key=agent_votes.count)
        results.append(most_common_solution)
    return results
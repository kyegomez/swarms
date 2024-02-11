import re
from collections import Counter


def extract_last_python_code_block(text):
    """
    Extracts the last Python code block from the given text.

    Args:
        text (str): The text to search for Python code blocks.

    Returns:
        str or None: The last Python code block found in the text, or None if no code block is found.
    """
    # The regular expression pattern for Python code blocks
    pattern = r"```[pP]ython(.*?)```"

    # Find all matches in the text
    matches = re.findall(pattern, text, re.DOTALL)

    # If there are matches, return the last one
    if matches:
        return matches[-1].strip()
    else:
        return None


def parse_code_completion(agent_response, question):
    """
    Parses the code completion response from the agent and extracts the last Python code block.

    Args:
        agent_response (str): The response from the agent.
        question (str): The original question.

    Returns:
        tuple: A tuple containing the parsed Python code and a boolean indicating success.
    """
    python_code = extract_last_python_code_block(agent_response)
    if python_code is None:
        if agent_response.count("impl]") == 0:
            python_code = agent_response
        else:
            python_code_lines = agent_response.split("\n")
            python_code = ""
            in_func = False
            for line in python_code_lines:
                if in_func:
                    python_code += line + "\n"
                if "impl]" in line:
                    in_func = True
    if python_code.count("def") == 0:
        python_code = question + python_code
    return python_code, True


def most_frequent(
    clist: list,
    cmp_func: callable = None,
):
    """
    Finds the most frequent element in a list based on a comparison function.

    Args:
        clist (list): The list of elements to search.
        cmp_func (function, optional): The comparison function used to determine the frequency of elements.
            If not provided, the default comparison function is used.

    Returns:
        tuple: A tuple containing the most frequent element and its frequency.
    """
    counter = 0
    num = clist[0]

    for i in clist:
        current_frequency = sum(cmp_func(i, item) for item in clist)
        print(current_frequency)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num, counter


def majority_voting(answers: list):
    """
    Performs majority voting on a list of answers and returns the most common answer.

    Args:
        answers (list): A list of answers.

    Returns:
        The most common answer in the list.
    """
    counter = Counter(answers)
    answer = counter.most_common(1)[0][0]
    return answer

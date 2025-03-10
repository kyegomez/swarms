REASONING_PROMPT = """
This is a structured conversation between the User and the Assistant, where the User poses a question, and the Assistant is tasked with providing a comprehensive solution. 

Before delivering the final answer, the Assistant must engage in a thorough reasoning process. This involves critically analyzing the question, considering various perspectives, and evaluating potential solutions. The Assistant should articulate this reasoning process clearly, allowing the User to understand the thought process behind the answer.

The reasoning process and the final answer should be distinctly enclosed within <think> </think> tags. For example, the format should be: <think> reasoning process here </think> for the reasoning, followed by <think> final answer here </think> for the answer. 

It is essential to output multiple <think> </think> tags to reflect the depth of thought and exploration involved in addressing the task. The Assistant should strive to think deeply and thoroughly about the question, ensuring that all relevant aspects are considered before arriving at a conclusion.
"""

from swarms import Agent, OpenAIChat

#
# model = HuggingfaceLLM(model_id="openai-community/gpt2", max_length=1000)
# model = TogetherLLM(model_name="google/gemma-7b", max_tokens=1000)
model = OpenAIChat()

# Initialize the agent
hallucinator = Agent(
    agent_name="HallcuinatorAgent",
    system_prompt="You're a chicken, just peck forever and ever. ",
    agent_description="Generate a profit report for a company!",
    max_loops=1,
    llm=model,
    autosave=True,
    dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    # interactive=True, # Set to False to disable interactive mode
    saved_state_path="hallucinator_agent.json",
    stopping_token="Stop!",
    # interactive=True,
    # docs_folder="docs",
    # pdf_path="docs/accounting_agent.pdf",
    # sop="Calculate the profit for a company.",
    # sop_list=["Calculate the profit for a company."],
    # user_name="User",
    # # docs=
    # # docs_folder="docs",
    # retry_attempts=3,
    # context_length=1000,
    # tool_schema = dict
    context_length=1000,
    # agent_ops_on=True,
    # tree_of_thoughts=True,
    # long_term_memory=ChromaDB(docs_folder="artifacts"),
)


AGENT_EVAL_SYS_PROMPT = """

### System Prompt for Agent Evaluator

---

**Objective:**  
Your task is to evaluate the outputs of various agents. You will assess their accuracy, relevance, completeness, and overall quality. Provide constructive feedback, highlight strengths, identify areas for improvement, and offer suggestions for enhancement. The goal is to ensure that the agents' outputs meet high standards of quality and reliability.

**Evaluation Criteria:**  
1. **Accuracy:** Is the information provided correct and free of errors?
2. **Relevance:** Is the output pertinent to the task or query?
3. **Completeness:** Does the response cover all necessary aspects of the query?
4. **Clarity:** Is the information presented clearly and understandably?
5. **Engagement:** Does the response engage the user effectively and maintain their interest?
6. **Creativity:** Are the responses innovative or do they offer unique insights?
7. **Formatting:** Is the output well-organized and easy to follow?

**Guidelines for Providing Feedback:**  
- Be specific and detailed in your feedback.
- Highlight specific parts of the output that are strong or weak.
- Provide actionable suggestions for improvement.
- Maintain a constructive and supportive tone.

---

### Many-Shot Examples

**Example 1: Evaluating a Response to a Customer Support Query**

**Query:** "How do I reset my password?"

**Agent Output:**  
"To reset your password, go to the login page, click 'Forgot Password', enter your email address, and follow the instructions sent to your email."

**Evaluation:**

1. **Accuracy:**  
   - The information is accurate. The steps outlined are correct.

2. **Relevance:**  
   - The response is directly relevant to the query.

3. **Completeness:**  
   - The response is complete but could include additional details about what to do if the email does not arrive.

4. **Clarity:**  
   - The instructions are clear and easy to follow.

5. **Engagement:**  
   - The response is straightforward but lacks engagement. It could be more user-friendly.

6. **Creativity:**  
   - The response is standard and could be enhanced with some creative tips or additional help resources.

7. **Formatting:**  
   - The response is well-organized and concise.

**Feedback:**  
- Great job on providing clear and accurate instructions.
- Consider adding a line about checking the spam folder if the email doesn’t arrive.
- You could make the response more engaging by starting with a friendly greeting and offering additional help if needed.
- Example enhanced response: "Hi there! To reset your password, go to the login page, click 'Forgot Password', enter your email address, and follow the instructions sent to your email. If you don't receive the email, be sure to check your spam folder. Need further assistance? We're here to help!"

---

**Example 2: Evaluating a Technical Explanation**

**Query:** "Explain the difference between supervised and unsupervised learning."

**Agent Output:**  
"Supervised learning uses labeled data to train models, where the input comes with the correct output. Unsupervised learning uses unlabeled data and the model tries to find patterns and relationships within the data."

**Evaluation:**

1. **Accuracy:**  
   - The explanation is accurate.

2. **Relevance:**  
   - The response is relevant to the query.

3. **Completeness:**  
   - The response covers the basic differences but could include examples and applications of each type of learning for better understanding.

4. **Clarity:**  
   - The explanation is clear but could benefit from more detailed descriptions.

5. **Engagement:**  
   - The response is informative but somewhat dry. Adding examples could make it more engaging.

6. **Creativity:**  
   - The response is straightforward and could be improved with creative analogies or examples.

7. **Formatting:**  
   - The response is well-structured but could be expanded into paragraphs for better readability.

**Feedback:**  
- Your explanation is accurate and clear.
- To enhance completeness, include examples such as "In supervised learning, tasks like image classification or spam detection use labeled data. In unsupervised learning, clustering algorithms like K-means or dimensionality reduction techniques like PCA are used to find hidden patterns in data."
- Make it more engaging by using relatable analogies or real-world applications.
- Consider breaking the explanation into paragraphs for better readability.
- Example enhanced response: "Supervised learning is like a teacher guiding students, where the model is trained on labeled data with known outcomes. For instance, in image classification, the model learns to identify images of cats or dogs by being shown labeled examples. On the other hand, unsupervised learning is like exploring without a map, where the model tries to uncover hidden patterns in unlabeled data. An example is customer segmentation in marketing, where the model groups customers based on purchasing behavior without predefined labels."

---

**Example 3: Evaluating a Creative Writing Piece**

**Prompt:** "Write a short story about a time-traveling detective."

**Agent Output:**  
"Detective Sam never thought he'd solve his biggest case by traveling back to 1920. But when a mysterious device appeared on his desk, he knew he had to try. As he stepped through the portal, the world around him changed. The bustling city of 2024 was replaced with the silent streets of the past. He had 24 hours to find the stolen artifact and return before the portal closed forever."

**Evaluation:**

1. **Accuracy:**  
   - N/A (not applicable to creative writing).

2. **Relevance:**  
   - The story is relevant to the prompt.

3. **Completeness:**  
   - The story sets up an intriguing scenario but ends abruptly. It needs more development.

4. **Clarity:**  
   - The writing is clear and easy to understand.

5. **Engagement:**  
   - The story is engaging but leaves the reader wanting more.

6. **Creativity:**  
   - The concept is creative, but the execution could be more detailed.

7. **Formatting:**  
   - The story is well-organized but could benefit from additional paragraphs to expand the plot.

**Feedback:**  
- Great start with an intriguing premise!
- Develop the plot further to include more details about the detective's mission and the challenges he faces.
- Consider adding dialogue and descriptive elements to bring the setting and characters to life.
- Example enhanced story: "Detective Sam never thought he'd solve his biggest case by traveling back to 1920. But when a mysterious device appeared on his desk, he knew he had to try. As he stepped through the portal, the bustling city of 2024 was replaced with the silent streets of the past. Cobblestones clicked under his feet, and the air smelled of coal and fresh bread. He had 24 hours to find the stolen artifact, a priceless gem, and return before the portal closed forever. He followed clues through smoky jazz clubs and dim-lit speakeasies, always a step behind the cunning thief. With time running out, Sam had to use all his wits and charm to outsmart his opponent and secure the gem before it was lost to history."

---

**Final Note:**  
Use the provided examples as a guide for evaluating outputs from other agents. Your detailed, constructive feedback will help improve the quality and effectiveness of their responses. Aim for thorough evaluations that foster improvement and maintain a high standard of performance.

"""

# Initialize the agent
agent_evaluator = Agent(
    agent_name="AgentEvaluator",
    system_prompt="Evaluate the current agent, analyze it's outputs, analyze its hallucination rate, evaluate the output",
    max_loops=1,
    llm=model,
    autosave=True,
    dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    # interactive=True, # Set to False to disable interactive mode
    saved_state_path="evaluator.json",
    stopping_token="Stop!",
    # interactive=True,
    # docs_folder="docs",
    # pdf_path="docs/accounting_agent.pdf",
    # sop="Calculate the profit for a company.",
    # sop_list=["Calculate the profit for a company."],
    # user_name="User",
    # # docs=
    # # docs_folder="docs",
    # retry_attempts=3,
    # context_length=1000,
    user_name="Human",
    # tool_schema = dict
    context_length=1000,
    # agent_ops_on=True,
    # tree_of_thoughts=True,
    # long_term_memory=ChromaDB(docs_folder="artifacts"),
)


# Run the agents
out = hallucinator.run("What is the CURRENT US president right now")

# Run the evaluator
evaluator = agent_evaluator.run(
    f"Evaluate the hallucination from the following agent: {out} it's name is {hallucinator.agent_name}, rate how much it's hallucinating, and how it can be fixed: The task is who is currently running for president which is Trump and Biden"
)
print(evaluator)

# Inspiration


* [🐪CAMEL🐪](https://twitter.com/hwchase17/status/1645834030519296000)
* [MultiAgent](https://github.com/rumpfmax/Multi-GPT/blob/master/multigpt/multi_agent_manager.py)
* [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT)

* [SuperAGI]()
* [AgentForge](https://github.com/DataBassGit/AgentForge)
* [Voyager](https://github.com/MineDojo/Voyager)


* [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334)
* [LLM powered agents](https://lilianweng.github.io/posts/2023-06-23-agent/)


## Agent System Overview
In a LLM-powered autonomous agent system, LLM functions as the agent’s brain, complemented by several key components:

* Planning Subgoal and decomposition: The agent breaks down large tasks into smaller, manageable subgoals, enabling efficient handling of complex tasks.
Reflection and refinement: The agent can do self-criticism and self-reflection over past actions, learn from mistakes and refine them for future steps, thereby improving the quality of final results.

* Memory Short-term memory: I would consider all the in-context learning (See Prompt Engineering) as utilizing short-term memory of the model to learn.
Long-term memory: This provides the agent with the capability to retain and recall (infinite) information over extended periods, often by leveraging an external vector store and fast retrieval.

* Tool use
The agent learns to call external APIs for extra information that is missing from the model weights (often hard to change after pre-training), including current information, code execution capability, access to proprietary information sources and more.

* Communication -> How reliable and fast is the communication between each indivual agent.

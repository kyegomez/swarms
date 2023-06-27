# Agora

![Agora banner](Agora-Banner-blend.png)

[Swarms is brought to you by Agora, the open source AI research organization. Join Agora and Help create swarms and or recieve support to advance Humanity. ](https://discord.gg/qUtxnK2NMf)

# Swarming Language Models (Swarms)
[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/swarms)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20AI%20project:%20&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms)

[![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms)

[![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&title=&summary=&source=)


Welcome to Swarms - the future of AI, where we leverage the power of autonomous agents to create 'swarms' of Language Models (LLM) that work together, creating a dynamic and interactive AI system.

## Vision
In the world of AI and machine learning, individual models have made significant strides in understanding and generating human-like text. But imagine the possibilities when these models are no longer solitary units, but part of a cooperative and communicative swarm. This is the future we envision.

Just as a swarm of bees works together, communicating and coordinating their actions for the betterment of the hive, swarming LLM agents can work together to create richer, more nuanced outputs. By harnessing the strengths of individual agents and combining them through a swarming architecture, we can unlock a new level of performance and responsiveness in AI systems. We envision swarms of LLM agents revolutionizing fields like customer support, content creation, research, and much more.


## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Sharing](#sharing)

## Installation
```bash
git clone https://github.com/kyegomez/swarms.git
cd swarms
pip install -r requirements.txt
```

## Usage

The primary agent in this repository is the `AutoAgent` from `./swarms/agents/workers/auto_agent.py`.

This `AutoAgent` is used to create the `MultiModalVisualAgent`, an autonomous agent that can process tasks in a multi-modal environment, like dealing with both text and visual data.

To use this agent, you need to import the agent and instantiate it. Here is a brief guide:

```python
from swarms.agents.auto_agent import MultiModalVisualAgent

# Initialize the agent
multimodal_agent = MultiModalVisualAgent()
```

### Working with MultiModalVisualAgentTool
The `MultiModalVisualAgentTool` class is a tool wrapper around the `MultiModalVisualAgent`. It simplifies working with the agent by encapsulating agent-related logic within its methods. Here's a brief guide on how to use it:

```python
from swarms.agents.auto_agent import MultiModalVisualAgent, MultiModalVisualAgentTool

# Initialize the agent
multimodal_agent = MultiModalVisualAgent()

# Initialize the tool with the agent
multimodal_agent_tool = MultiModalVisualAgentTool(multimodal_agent)

# Now, you can use the agent tool to perform tasks. The run method is one of them.
result = multimodal_agent_tool.run('Your text here')
```

## Note
- The `AutoAgent` makes use of several helper tools and context managers for tasks such as processing CSV files, browsing web pages, and querying web pages. For the best use of this agent, understanding these tools is crucial.

- Additionally, the agent uses the ChatOpenAI, a language learning model (LLM), to perform its tasks. You need to provide an OpenAI API key to make use of it. 

- Detailed knowledge of FAISS, a library for efficient similarity search and clustering of dense vectors, is also essential as it's used for memory storage and retrieval. 


## Swarming Architectures

Here are three examples of swarming architectures that could be applied in this context.

1. **Hierarchical Swarms**: In this architecture, a 'lead' agent coordinates the efforts of other agents, distributing tasks based on each agent's unique strengths. The lead agent might be equipped with additional functionality or decision-making capabilities to effectively manage the swarm.

2. **Collaborative Swarms**: Here, each agent in the swarm works in parallel, potentially on different aspects of a task. They then collectively determine the best output, often through a voting or consensus mechanism.

3. **Competitive Swarms**: In this setup, multiple agents work on the same task independently. The output from the agent which produces the highest confidence or quality result is then selected. This can often lead to more robust outputs, as the competition drives each agent to perform at its best.

4. **Multi-Agent Debate**: Here, multiple agents debate a topic. The output from the agent which produces the highest confidence or quality result is then selected. This can lead to more robust outputs, as the competition drives each agent to perform it's best.


## Share with your Friends

If you love what we're building here, please consider sharing our project with your friends and colleagues! You can use the following buttons to share on social media.

[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/swarms)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20AI%20project:%20&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms)

[![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms)

[![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&title=&summary=&source=)

## Contribute
We're always looking for contributors to help us improve and expand this project. If you're interested, please check out our [Contributing Guidelines](./CONTRIBUTING.md).

Thank you for being a part of our project!

# Ideas

A swarm, particularly in the context of distributed computing, refers to a large number of coordinated agents or nodes that work together to solve a problem. The specific requirements of a swarm might vary depending on the task at hand, but some of the general requirements include:

1. **Distributed Nature**: The swarm should consist of multiple individual units or nodes, each capable of functioning independently.

2. **Coordination**: The nodes in the swarm need to coordinate with each other to ensure they're working together effectively. This might involve communication between nodes, or it could be achieved through a central orchestrator.

3. **Scalability**: A well-designed swarm system should be able to scale up or down as needed, adding or removing nodes based on the task load.

4. **Resilience**: If a node in the swarm fails, it shouldn't bring down the entire system. Instead, other nodes should be able to pick up the slack.

5. **Load Balancing**: Tasks should be distributed evenly across the nodes in the swarm to avoid overloading any single node.

6. **Interoperability**: Each node should be able to interact with others, regardless of differences in underlying hardware or software.

Integrating these requirements with Large Language Models (LLMs) can be done as follows:

1. **Distributed Nature**: Each LLM agent can be considered as a node in the swarm. These agents can be distributed across multiple servers or even geographically dispersed data centers.

2. **Coordination**: An orchestrator can manage the LLM agents, assigning tasks, coordinating responses, and ensuring effective collaboration between agents.

3. **Scalability**: As the demand for processing power increases or decreases, the number of LLM agents can be adjusted accordingly. 

4. **Resilience**: If an LLM agent goes offline or fails, the orchestrator can assign its tasks to other agents, ensuring the swarm continues functioning smoothly.

5. **Load Balancing**: The orchestrator can also handle load balancing, ensuring tasks are evenly distributed amongst the LLM agents.

6. **Interoperability**: By standardizing the input and output formats of the LLM agents, they can effectively communicate and collaborate, regardless of the specific model or configuration of each agent.

In terms of architecture, the swarm might look something like this:

```
                                           (Orchestrator)
                                             /        \
           (LLM Agent)---(Communication Layer)       (Communication Layer)---(LLM Agent)
              /                  |                                           |                 \
(Task Assignment)      (Task Completion)                    (Task Assignment)       (Task Completion)
```

Each LLM agent communicates with the orchestrator through a dedicated communication layer. The orchestrator assigns tasks to each LLM agent, which the agents then complete and return. This setup allows for a high degree of flexibility, scalability, and robustness.



# To do:

* Integrate [Multi Agent debate](https://github.com/Skytliang/Multi-Agents-Debate)

* Integrate [Multi agent2 debate](https://github.com/composable-models/llm_multiagent_debate)

* Integrate meta prompting into all worker agents

* Create 1 main swarms class `swarms('Increase sales by 40$', workers=4)`

* Integrate [Jarvis](https://github.com/microsoft/JARVIS) as worker nodes

* Integrate guidance and token healing
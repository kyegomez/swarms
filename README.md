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
There are 2 methods, one is very simple to test it out and then there is another to explore the agents and so on!

### Method 1
Simple example `python3 example.py`

### Method 2
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


## Share with your Friends

Share on Twitter: [![Share on Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms)](https://twitter.com/intent/tweet?text=Check%20out%20Swarms%20-%20the%20future%20of%20AI%20%23swarms%20%23AI&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms)

Share on Facebook: [![Share on Facebook](https://img.shields.io/badge/-Share%20on%20Facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms)

Share on LinkedIn: [![Share on LinkedIn](https://img.shields.io/badge/-Share%20on%20LinkedIn-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&title=Swarms%20-%20the%20future%20of%20AI&summary=Check%20out%20Swarms%2C%20the%20future%20of%20AI%20where%20swarms%20of%20Language%20Models%20work%20together%20to%20create%20dynamic%20and%20interactive%20AI%20systems.&source=)

Share on Reddit: [![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&title=Swarms%20-%20the%20future%20of%20AI)

Share on Hacker News: [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&t=Swarms%20-%20the%20future%20of%20AI)

Share on Pinterest: [![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=Swarms%20-%20the%20future%20of%20AI)

Share on WhatsApp: [![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=Check%20out%20Swarms%20-%20the%20future%20of%20AI%20%23swarms%20%23AI%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms)

## Contribute
We're always looking for contributors to help us improve and expand this project. If you're interested, please check out our [Contributing Guidelines](./CONTRIBUTING.md).

Thank you for being a part of our project!


# To do:

* Integrate [Multi Agent debate](https://github.com/Skytliang/Multi-Agents-Debate)

* Integrate [Multi agent2 debate](https://github.com/composable-models/llm_multiagent_debate)

* Integrate meta prompting into all worker agents

* Create 1 main swarms class `swarms('Increase sales by 40$', workers=4)`

* Integrate [Jarvis](https://github.com/microsoft/JARVIS) as worker nodes

* Integrate guidance and token healing

* Add text to speech [whisper x, youtube script](https://github.com/kyegomez/youtubeURL-to-text) and text to speech code models as tools 

* Add task completion logic with meta prompting, task evaluation as a state from 0.0 to 1.0, and critiquing for meta prompting.

* Integrate meta prompting for every agent boss and worker

* Get baby agi set up with the autogpt instance as a tool

* Integrate [Ocean](https://github.com/kyegomez/Ocean) vector db as the main embedding database for all the agents boss and or worker

* Communication, a universal vector database that is only used when a task is completed in this format `[TASK][COMPLETED]`

* Create unit tests

* Create benchmrks

* Create evaluations

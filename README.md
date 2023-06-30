# Agora

![Agora banner](Agora-Banner-blend.png)

[Swarms is brought to you by Agora, the open source AI research organization. Join Agora and Help create swarms and or recieve support to advance Humanity. ](https://discord.gg/qUtxnK2NMf)

# Swarming Language Models (Swarms)

![Swarming banner](swarms.png)

[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/swarms)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20AI%20project:%20&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms)

[![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms)

[![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&title=&summary=&source=)


Welcome to Swarms - the future of AI, where we leverage the power of autonomous agents to create 'swarms' of Language Models (LLM) that work together, creating a dynamic and interactive AI system.

## Vision
In the world of AI and machine learning, individual models have made significant strides in understanding and generating human-like text. But imagine the possibilities when these models are no longer solitary units, but part of a cooperative and communicative swarm. This is the future we envision.

Just as a swarm of bees works together, communicating and coordinating their actions for the betterment of the hive, swarming LLM agents can work together to create richer, more nuanced outputs. By harnessing the strengths of individual agents and combining them through a swarming architecture, we can unlock a new level of performance and responsiveness in AI systems. We envision swarms of LLM agents revolutionizing fields like customer support, content creation, research, and much more.


## Table of Contents
1. [Installation](#installation)
2. [Sharing](#sharing)

## Installation
There are 2 methods, one is through `git clone` and the other is by `pip install swarms`. Check out the [document](/DOCUMENTATION.md) for more information on the classes.

# Method1
* Pip install `python3 -m pip install swarms`

* Create new python file and unleash superintelligence

```python

from swarms import boss_node

#create a task
task = boss_node.create_task(objective="Write a research paper on the impact of climate change on global agriculture")

#or 
# task = boss_node.create_task('Find a video of Elon Musk and make him look like a cat')

boss_node.execute(task)
```

# Method2
Download via Github, and install requirements
```bash
git clone https://github.com/kyegomez/swarms.git
cd swarms
pip install -r requirements.txt
```

### Method 3
Simple example by `git cloning https://github.com/kyegomez/swarms.git` `python3 example.py`

## BossNode

The `BossNode` class is a key component of Swarms. It represents a "boss" in the system that assigns tasks to other components.

Here is an example of how you can use it:

```python
class BossNode:
    def __init__(self, tools):
        # initialization code goes here

    def create_task(self, objective):
        return {"objective": objective}

    def execute_task(self, task):
        # task execution code goes here
```

With the `BossNode` class, you can create tasks for your tools to perform. For example, you can create a task to write a summary of a specific topic:

```python
from swarms import boss_node
#create a task
task = boss_node.create_task(objective="Write a research paper on the impact of climate change on global agriculture")
#execute the teask
boss_node.execute_task(task)

```

This will create and execute a task to write a summary about the latest news on quantum computing. The result will be the summary of the news.


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

* Add new tool that uses WhiseperX to transcribe a youtube video

* Integrate hf agents as tools

* [Integrate tools from here](https://integrations.langchain.com/)


* Create extensive and useful examples 

* And, recreate exampels and usage in readme.

* Create a worker Swarm class, where it's just workers who are equal and that can self scale. If they need help they'll just spawn an entirely new worker and they can spawn more workers





# Optimization

* Reliability => The swarm needs to be reliable. How do we quantify reliability -> Reliability is obtaining an desired output with a basic and un-detailed input. 

* Speed => How long does it take the swarm to accomplish a task, such as `let's respond to all the emails`, we need to minimize this => we can do this by cultivating an efficient communication layer, critiquing, and self-alignment with meta prompting.

* Scalability => Asynchrony, Concurrent, and self-healing.
![Swarming banner icon](images/swarmsbanner.png)

<div align="center">

Swarms is a modular framework that enables reliable and useful multi-agent collaboration at scale to automate real-world tasks.


[![GitHub issues](https://img.shields.io/github/issues/kyegomez/swarms)](https://github.com/kyegomez/swarms/issues) [![GitHub forks](https://img.shields.io/github/forks/kyegomez/swarms)](https://github.com/kyegomez/swarms/network) [![GitHub stars](https://img.shields.io/github/stars/kyegomez/swarms)](https://github.com/kyegomez/swarms/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/swarms)](https://github.com/kyegomez/swarms/blob/main/LICENSE)[![GitHub star chart](https://img.shields.io/github/stars/kyegomez/swarms?style=social)](https://star-history.com/#kyegomez/swarms)[![Dependency Status](https://img.shields.io/librariesio/github/kyegomez/swarms)](https://libraries.io/github/kyegomez/swarms) [![Downloads](https://static.pepy.tech/badge/swarms/month)](https://pepy.tech/project/swarms)


### Share on Social Media

[![Join the Agora discord](https://img.shields.io/discord/1110910277110743103?label=Discord&logo=discord&logoColor=white&style=plastic&color=d7b023)![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/swarms)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20AI%20project:%20&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms) [![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms) [![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&title=&summary=&source=)

[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&title=Swarms%20-%20the%20future%20of%20AI) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&t=Swarms%20-%20the%20future%20of%20AI) [![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=Swarms%20-%20the%20future%20of%20AI) [![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=Check%20out%20Swarms%20-%20the%20future%20of%20AI%20%23swarms%20%23AI%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms)

</div>


## Purpose
At Swarms, we're transforming the landscape of AI from siloed AI agents to a unified 'swarm' of intelligence. Through relentless iteration and the power of collective insight from our 1500+ Agora researchers, we're developing a groundbreaking framework for AI collaboration. Our mission is to catalyze a paradigm shift, advancing Humanity with the power of unified autonomous AI agent swarms.

-----

# 🤝 Schedule a 1-on-1 Session
Book a [1-on-1 Session with Kye](https://calendly.com/apacai/agora), the Creator, to discuss any issues, provide feedback, or explore how we can improve Swarms for you.


----------

## Installation
* `pip3 install swarms`

---

## Usage
We have a small gallery of examples to run here, [for more check out the docs to build your own agent and or swarms!](https://docs.apac.ai)

### `MultiAgentDebate`

- `MultiAgentDebate` is a simple class that enables multi agent collaboration.

```python
from swarms.workers import Worker
from swarms.swarms import MultiAgentDebate, select_speaker
from swarms.models import OpenAIChat

llm = OpenAIChat(
    model_name='gpt-4', 
    openai_api_key="api-key", 
    temperature=0.5
)

node = Worker(
    llm=llm,
    ai_name="Optimus Prime",
    ai_role="Worker in a swarm",
    external_tools = None,
    human_in_the_loop = False,
    temperature = 0.5,
)

node2 = Worker(
    llm=llm,
    ai_name="Bumble Bee",
    ai_role="Worker in a swarm",
    external_tools = None,
    human_in_the_loop = False,
    temperature = 0.5,
)

node3 = Worker(
    llm=llm,
    ai_name="Bumble Bee",
    ai_role="Worker in a swarm",
    external_tools = None,
    human_in_the_loop = False,
    temperature = 0.5,
)

agents = [
    node,
    node2,
    node3
]

# Initialize multi-agent debate with the selection function
debate = MultiAgentDebate(agents, select_speaker)

# Run task
task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
results = debate.run(task, max_iters=4)

# Print results
for result in results:
    print(f"Agent {result['agent']} responded: {result['response']}")
```

----

### `Worker`
- The `Worker` is an fully feature complete agent with an llm, tools, and a vectorstore for long term memory!
- Place your api key as parameters in the llm if you choose!
- And, then place the openai api key in the Worker for the openai embedding model

```python
from swarms.models import ChatOpenAI
from swarms.workers import Worker

llm = ChatOpenAI(
    model_name='gpt-4', 
    openai_api_key="api-key", 
    temperature=0.5
)

node = Worker(
    llm=llm,
    ai_name="Optimus Prime",
    #openai key for the embeddings
    openai_api_key="sk-eee"
    ai_role="Worker in a swarm",
    external_tools = None,
    human_in_the_loop = False,
    temperature = 0.5,
)

task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
response = node.run(task)
print(response)

```

------

### `OmniModalAgent`
- OmniModal Agent is an LLM that access to 10+ multi-modal encoders and diffusers! It can generate images, videos, speech, music and so much more, get started with:

```python
from swarms.models import OpenAIChat
from swarms.agents import OmniModalAgent


llm = OpenAIChat(model_name="gpt-4")

agent = OmniModalAgent(llm)

agent.run("Create a video of a swarm of fish")

```

- OmniModal Agent has a ui in the root called `python3 omni_ui.py`

---

# Documentation
For documentation, go here, [swarms.apac.ai](https://swarms.apac.ai)

-----

# ✨ Features
* Easy to use Base LLMs, `OpenAI` `Palm` `Anthropic` `HuggingFace`
* Enterprise Grade, Production Ready with robust Error Handling
* Multi-Modality Native with Multi-Modal LLMs as tools
* Infinite Memory Processing: Store infinite sequences of infinite Multi-Modal data, text, images, videos, audio
* Usability: Extreme emphasis on useability, code is at it's theortical minimum simplicity factor to use
* Reliability: Outputs that accomplish tasks and activities you wish to execute.
* Fluidity: A seamless all-around experience to build production grade workflows
* Speed: Lower the time to automate tasks by 90%. 
* Simplicity: Swarms is extremely simple to use, if not thee simplest agent framework of all time
* Powerful: Swarms is capable of building entire software apps, to large scale data analysis, and handling chaotic situations


-----

## Contribute
We're always looking for contributors to help us improve and expand this project. If you're interested, please check out our [Contributing Guidelines](C0NTRIBUTING.md).



### Optimization Priorities

1. **Reliability**: Increase the reliability of the swarm - obtaining the desired output with a basic and un-detailed input.

2. **Speed**: Reduce the time it takes for the swarm to accomplish tasks by improving the communication layer, critiquing, and self-alignment with meta prompting.

3. **Scalability**: Ensure that the system is asynchronous, concurrent, and self-healing to support scalability.

Our goal is to continuously improve Swarms by following this roadmap, while also being adaptable to new needs and opportunities as they arise.

---

# Demos

![Swarms Demo](images/Screenshot_48.png)

## Swarm Video Demo {Click for more}

[![Watch the swarm video](https://img.youtube.com/vi/Br62cDMYXgc/maxresdefault.jpg)](https://youtu.be/Br62cDMYXgc)

---

# Contact 
For enterprise and production ready deployments, allow us to discover more about you and your story, [book a call with us here](https://www.apac.ai/Setup-Call)
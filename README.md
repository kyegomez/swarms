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


## Hiring
We're hiring: Engineers, Researchers, Interns And, salesprofessionals to work on democratizing swarms, email me at with your story at `kye@apac.ai`

----------

## Installation

There are 2 methods, one is through `git clone` and the other is by `pip install swarms`. Check out the [DOCUMENTATION](DOCS/DOCUMENTATION.md) for more information on the classes.

* Pip install `pip3 install swarms`

* Create new python file and unleash superintelligence

```python

from swarms import Worker


node = Worker(
    openai_api_key="",
    ai_name="Optimus Prime",
)

task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
response = node.run(task)
print(response)
```

---

## Usage

```python

from swarms import HuggingFaceLLM

hugging_face_model = HuggingFaceLLM(model_id="Voicelab/trurl-2-13b")
generated_text = hugging_face_model.generate("In a world where AI")

```
```python

from swarms import Worker


node = Worker(
    openai_api_key="",
    ai_name="Optimus Prime",
)

task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
response = node.run(task)
print(response)

```
---

# Documentation
For documentation, go here, [swarms.apac.ai](https://swarms.apac.ai)

**NOTE: We need help building the documentation**

-----

# Docker Setup
The docker file is located in the docker folder in the `infra` folder, [click here and navigate here in your environment](/infra/Docker)

* Build the Docker image

* You can build the Docker image using the provided Dockerfile. Navigate to the infra/Docker directory where the Dockerfiles are located.

* For the CPU version, use:

```bash
docker build -t swarms-api:latest -f Dockerfile.cpu .
```
For the GPU version, use:

```bash
docker build -t swarms-api:gpu -f Dockerfile.gpu .
```
### Run the Docker container

After building the Docker image, you can run the Swarms API in a Docker container. Replace your_redis_host and your_redis_port with your actual Redis host and port.

For the CPU version:

```bash
docker run -p 8000:8000 -e REDIS_HOST=your_redis_host -e REDIS_PORT=your_redis_port swarms-api:latest
```

## For the GPU version:
```bash
docker run --gpus all -p 8000:8000 -e REDIS_HOST=your_redis_host -e REDIS_PORT=your_redis_port swarms-api:gpu
```

## Access the Swarms API

* The Swarms API will be accessible at http://localhost:8000. You can use tools like curl or Postman to send requests to the API.

Here's an example curl command to send a POST request to the /chat endpoint:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"api_key": "your_openai_api_key", "objective": "your_objective"}' http://localhost:8000/chat
```
Replace your_openai_api_key and your_objective with your actual OpenAI API key and objective.

----


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
We're always looking for contributors to help us improve and expand this project. If you're interested, please check out our [Contributing Guidelines](DOCS/C0NTRIBUTING.md).

Thank you for being a part of our project!

---
# Roadmap

Please checkout our [Roadmap](DOCS/ROADMAP.md) and consider contributing to make the dream of Swarms real to advance Humanity.

## Optimization Priorities

1. **Reliability**: Increase the reliability of the swarm - obtaining the desired output with a basic and un-detailed input.

2. **Speed**: Reduce the time it takes for the swarm to accomplish tasks by improving the communication layer, critiquing, and self-alignment with meta prompting.

3. **Scalability**: Ensure that the system is asynchronous, concurrent, and self-healing to support scalability.

Our goal is to continuously improve Swarms by following this roadmap, while also being adaptable to new needs and opportunities as they arise.

---

# Bounty Program

Our bounty program is an exciting opportunity for contributors to help us build the future of Swarms. By participating, you can earn rewards while contributing to a project that aims to revolutionize digital activity.

Here's how it works:

1. **Check out our Roadmap**: We've shared our roadmap detailing our short and long-term goals. These are the areas where we're seeking contributions.

2. **Pick a Task**: Choose a task from the roadmap that aligns with your skills and interests. If you're unsure, you can reach out to our team for guidance.

3. **Get to Work**: Once you've chosen a task, start working on it. Remember, quality is key. We're looking for contributions that truly make a difference.

4. **Submit your Contribution**: Once your work is complete, submit it for review. We'll evaluate your contribution based on its quality, relevance, and the value it brings to Swarms.

5. **Earn Rewards**: If your contribution is approved, you'll earn a bounty. The amount of the bounty depends on the complexity of the task, the quality of your work, and the value it brings to Swarms.

---

## The Plan

### Phase 1: Building the Foundation
In the first phase, our focus is on building the basic infrastructure of Swarms. This includes developing key components like the Swarms class, integrating essential tools, and establishing task completion and evaluation logic. We'll also start developing our testing and evaluation framework during this phase. If you're interested in foundational work and have a knack for building robust, scalable systems, this phase is for you.

### Phase 2: Optimizing the System
In the second phase, we'll focus on optimizng Swarms by integrating more advanced features, improving the system's efficiency, and refining our testing and evaluation framework. This phase involves more complex tasks, so if you enjoy tackling challenging problems and contributing to the development of innovative features, this is the phase for you.

### Phase 3: Towards Super-Intelligence
The third phase of our bounty program is the most exciting - this is where we aim to achieve super-intelligence. In this phase, we'll be working on improving the swarm's capabilities, expanding its skills, and fine-tuning the system based on real-world testing and feedback. If you're excited about the future of AI and want to contribute to a project that could potentially transform the digital world, this is the phase for you.

Remember, our roadmap is a guide, and we encourage you to bring your own ideas and creativity to the table. We believe that every contribution, no matter how small, can make a difference. So join us on this exciting journey and help us create the future of Swarms.

<!-- **To participate in our bounty program, visit the [Swarms Bounty Program Page](https://swarms.ai/bounty).** Let's build the future together! -->



---

# EcoSystem

* [The-Compiler, compile natural language into serene, reliable, and secure programs](https://github.com/kyegomez/the-compiler)

*[The Replicator, an autonomous swarm that conducts Multi-Modal AI research by creating new underlying mathematical operations and models](https://github.com/kyegomez/The-Replicator)

* Make a swarm that checks arxviv for papers -> checks if there is a github link -> then implements them and checks them

* [SwarmLogic, where a swarm is your API, database, and backend!](https://github.com/kyegomez/SwarmLogic)

---

# Demos

![Swarms Demo](images/Screenshot_48.png)

## Swarm Video Demo {Click for more}

[![Watch the swarm video](https://img.youtube.com/vi/Br62cDMYXgc/maxresdefault.jpg)](https://youtu.be/Br62cDMYXgc)

---

# Contact 
For enterprise and production ready deployments, allow us to discover more about you and your story, [book a call with us here](https://www.apac.ai/Setup-Call)
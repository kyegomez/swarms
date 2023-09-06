The Swarms framework provides developers with the ability to create AI systems that operate across two dimensions: **predictability** and **creativity**. 

For **predictability**, Swarms enforces structures like sequential pipelines, DAG-based workflows, and long-term memory. To facilitate creativity, Swarms safely prompts LLMs with tools and short-term memory connecting them to external APIs and data stores. The framework allows developers to transition between those two dimensions effortlessly based on their use case.

Swarms not only helps developers harness the potential of LLMs but also enforces trust boundaries, schema validation, and tool activity-level permissions. By doing so, Swarms maximizes LLMs’ reasoning while adhering to strict policies regarding their capabilities.

Swarms’s design philosophy is based on the following tenets:

1. **Modularity and composability**: All framework primitives are useful and usable on their own in addition to being easy to plug into each other.
2. **Technology-agnostic**: Swarms is designed to work with any capable LLM, data store, and backend through the abstraction of drivers.
3. **Keep data off prompt by default**: When working with data through loaders and tools, Swarms aims to keep it off prompt by default, making it easy to work with big data securely and with low latency.
4. **Minimal prompt engineering**: It’s much easier to reason about code written in Python, not natural languages. Swarms aims to default to Python in most cases unless absolutely necessary.


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
---

# Documentation
For documentation, go here, [the docs folder in the root diectory](https://swarms.apac.ai)

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
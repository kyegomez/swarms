![Swarming banner icon](images/swarmslogobanner.png)

<div align="center">

Swarms is a modular framework that enables reliable and useful multi-agent collaboration at scale to automate real-world tasks.


[![GitHub issues](https://img.shields.io/github/issues/kyegomez/swarms)](https://github.com/kyegomez/swarms/issues) [![GitHub forks](https://img.shields.io/github/forks/kyegomez/swarms)](https://github.com/kyegomez/swarms/network) [![GitHub stars](https://img.shields.io/github/stars/kyegomez/swarms)](https://github.com/kyegomez/swarms/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/swarms)](https://github.com/kyegomez/swarms/blob/main/LICENSE)[![GitHub star chart](https://img.shields.io/github/stars/kyegomez/swarms?style=social)](https://star-history.com/#kyegomez/swarms)[![Dependency Status](https://img.shields.io/librariesio/github/kyegomez/swarms)](https://libraries.io/github/kyegomez/swarms) [![Downloads](https://static.pepy.tech/badge/swarms/month)](https://pepy.tech/project/swarms)

[![Join the Agora discord](https://img.shields.io/discord/1110910277110743103?label=Discord&logo=discord&logoColor=white&style=plastic&color=d7b023)![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/swarms)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20AI%20project:%20&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms) [![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms) [![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&title=&summary=&source=)

[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&title=Swarms%20-%20the%20future%20of%20AI) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&t=Swarms%20-%20the%20future%20of%20AI) [![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=Swarms%20-%20the%20future%20of%20AI) [![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=Check%20out%20Swarms%20-%20the%20future%20of%20AI%20%23swarms%20%23AI%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms)

</div>


----

## Installation
`pip3 install --upgrade swarms`

---

## Usage

Run example in Collab: <a target="_blank" href="https://colab.research.google.com/github/kyegomez/swarms/blob/master/playground/swarms_example.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### `Agent` Example
- Reliable Structure that provides LLMS autonomy
- Extremely Customizeable with stopping conditions, interactivity, dynamical temperature, loop intervals, and so much more
- Enterprise Grade + Production Grade: `Agent` is designed and optimized for automating real-world tasks at scale!

```python
import os

from dotenv import load_dotenv

# Import the OpenAIChat model and the Agent struct
from swarms.models import OpenAIChat
from swarms.structs import Agent

# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the language model
llm = OpenAIChat(
    temperature=0.5,
    model_name="gpt-4",
    openai_api_key=api_key,
    max_tokens=4000
)


## Initialize the workflow
agent = Agent(llm=llm, max_loops=1, autosave=True, dashboard=True)

# Run the workflow on a task
agent.run("Generate a 10,000 word blog on health and wellness.")



```

------

### `SequentialWorkflow`
- A Sequential swarm of autonomous agents where each agent's outputs are fed into the next agent
- Save and Restore Workflow states!
- Integrate Agent's with various LLMs and Multi-Modality Models

```python
import os 
from swarms.models import OpenAIChat
from swarms.structs import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow
from dotenv import load_dotenv

load_dotenv()

# Load the environment variables
api_key = os.getenv("OPENAI_API_KEY")


# Initialize the language agent
llm = OpenAIChat(
    temperature=0.5,
    model_name="gpt-4",
    openai_api_key=api_key,
    max_tokens=4000
)


# Initialize the agent with the language agent
agent1 = Agent(llm=llm, max_loops=1)

# Create another agent for a different task
agent2 = Agent(llm=llm, max_loops=1)

# Create another agent for a different task
agent3 = Agent(llm=llm, max_loops=1)

# Create the workflow
workflow = SequentialWorkflow(max_loops=1)

# Add tasks to the workflow
workflow.add(
    agent1, "Generate a 10,000 word blog on health and wellness.", 
)

# Suppose the next task takes the output of the first task as input
workflow.add(
    agent2, "Summarize the generated blog",
)

# Run the workflow
workflow.run()

# Output the results
for task in workflow.tasks:
    print(f"Task: {task.description}, Result: {task.result}")


```

## `Multi Modal Autonomous Agents`
- Run the agent with multiple modalities useful for various real-world tasks in manufacturing, logistics, and health.

```python
# Description: This is an example of how to use the Agent class to run a multi-modal workflow
import os
from dotenv import load_dotenv
from swarms.models.gpt4_vision_api import GPT4VisionAPI
from swarms.structs import Agent

# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the language model
llm = GPT4VisionAPI(
    openai_api_key=api_key,
    max_tokens=500,
)

# Initialize the task
task = (
    "Analyze this image of an assembly line and identify any issues such as"
    " misaligned parts, defects, or deviations from the standard assembly"
    " process. IF there is anything unsafe in the image, explain why it is"
    " unsafe and how it could be improved."
)
img = "assembly_line.jpg"

## Initialize the workflow
agent = Agent(
    llm=llm,
    max_loops="auto",
    autosave=True,
    dashboard=True,
    multi_modal=True
)

# Run the workflow on a task
agent.run(task=task, img=img)


```


### `OmniModalAgent`
- An agent that can understand any modality and conditionally generate any modality.

```python
from swarms.agents.omni_modal_agent import OmniModalAgent, OpenAIChat
from swarms.models import OpenAIChat
from dotenv import load_dotenv
import os

# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the language model
llm = OpenAIChat(
    temperature=0.5,
    model_name="gpt-4",
    openai_api_key=api_key,
)


agent = OmniModalAgent(llm)
agent.run("Generate a video of a swarm of fish and then make an image out of the video")
```

---

### Multi-Agent Swarm for Logistics
- Swarms is a framework designed for real-world deployment here is a demo presenting a fully ready to use Swarm for a vast array of logistics tasks.
- Swarms is designed to be modular and reliable for real-world deployments.
- Swarms is the first framework that unleases multi-modal autonomous agents in the real world.

```python
from swarms.structs import Agent
import os
from dotenv import load_dotenv
from swarms.models import GPT4VisionAPI
from swarms.prompts.logistics import (
    Health_Security_Agent_Prompt,
    Quality_Control_Agent_Prompt,
    Productivity_Agent_Prompt,
    Safety_Agent_Prompt,
    Security_Agent_Prompt,
    Sustainability_Agent_Prompt,
    Efficiency_Agent_Prompt,
)

# Load ENV
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# GPT4VisionAPI 
llm = GPT4VisionAPI(openai_api_key=api_key)

# Image for analysis
factory_image = "factory_image1.jpg"

# Initialize agents with respective prompts
health_security_agent = Agent(
    llm=llm,
    sop=Health_Security_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)

# Quality control agent
quality_control_agent = Agent(
    llm=llm,
    sop=Quality_Control_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)


# Productivity Agent
productivity_agent = Agent(
    llm=llm,
    sop=Productivity_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)

# Initiailize safety agent
safety_agent = Agent(
    llm=llm, sop=Safety_Agent_Prompt, max_loops=1, multi_modal=True
)

# Init the security agent
security_agent = Agent(
    llm=llm, sop=Security_Agent_Prompt, max_loops=1, multi_modal=True
)


# Initialize sustainability agent
sustainability_agent = Agent(
    llm=llm,
    sop=Sustainability_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)


# Initialize efficincy agent
efficiency_agent = Agent(
    llm=llm,
    sop=Efficiency_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)

# Run agents with respective tasks on the same image
health_analysis = health_security_agent.run(
    "Analyze the safety of this factory", factory_image
)
quality_analysis = quality_control_agent.run(
    "Examine product quality in the factory", factory_image
)
productivity_analysis = productivity_agent.run(
    "Evaluate factory productivity", factory_image
)
safety_analysis = safety_agent.run(
    "Inspect the factory's adherence to safety standards",
    factory_image,
)
security_analysis = security_agent.run(
    "Assess the factory's security measures and systems",
    factory_image,
)
sustainability_analysis = sustainability_agent.run(
    "Examine the factory's sustainability practices", factory_image
)
efficiency_analysis = efficiency_agent.run(
    "Analyze the efficiency of the factory's manufacturing process",
    factory_image,
)
```

### `Gemini`
- Deploy Gemini from Google with utmost reliability with our visual chain of thought prompt that enables more reliable responses
```python
import os

from dotenv import load_dotenv

from swarms.models import Gemini
from swarms.prompts.visual_cot import VISUAL_CHAIN_OF_THOUGHT

# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("GEMINI_API_KEY")

# Initialize the language model
llm = Gemini(
    gemini_api_key=api_key,
    temperature=0.5,
    max_tokens=1000,
    system_prompt=VISUAL_CHAIN_OF_THOUGHT,
)

# Initialize the task
task = "This is an eye test. What do you see?"
img = "playground/demos/multi_modal_chain_of_thought/eyetest.jpg"

# Run the workflow on a task
out = llm.run(task=task, img=img)
print(out)
```


### `Anthropic`
```python
# Import necessary modules and classes
from swarms.models import Anthropic

# Initialize an instance of the Anthropic class
model = Anthropic(
    anthropic_api_key=""
)

# Using the run method
completion_1 = model.run("What is the capital of France?")
print(completion_1)

# Using the __call__ method
completion_2 = model("How far is the moon from the earth?", stop=["miles", "km"])
print(completion_2)

```


### `HuggingFaceLLM`
```python
from swarms.models import HuggingfaceLLM

# Initialize with custom configuration
custom_config = {
    "quantize": True,
    "quantization_config": {"load_in_4bit": True},
    "verbose": True
}
inference = HuggingfaceLLM(model_id="NousResearch/Nous-Hermes-2-Vision-Alpha", **custom_config)

# Generate text based on a prompt
prompt_text = "Create a list of known biggest risks of structural collapse with references"
generated_text = inference(prompt_text)
print(generated_text)
```

### `Mixtral`
- Utilize Mixtral in a very simple API,
- Utilize 4bit quantization for a increased speed and less memory usage
- Use Flash Attention 2.0 for increased speed and less memory usage
```python
from swarms.models import Mixtral

# Initialize the Mixtral model with 4 bit and flash attention!
mixtral = Mixtral(load_in_4bit=True, use_flash_attention_2=True)

# Generate text for a simple task
generated_text = mixtral.run("Generate a creative story.")

# Print the generated text
print(generated_text)
```


### `Dalle3`
```python
from swarms import Dalle3

# Create an instance of the Dalle3 class with high quality
dalle3 = Dalle3(quality="high")

# Define a text prompt
task = "A high-quality image of a sunset"

# Generate a high-quality image from the text prompt
image_url = dalle3(task)

# Print the generated image URL
print(image_url)
```


### `GPT4Vision`
```python
from swarms.models import GPT4VisionAPI

# Initialize with default API key and custom max_tokens
api = GPT4VisionAPI(max_tokens=1000)

# Define the task and image URL
task = "Describe the scene in the image."
img = "https://i.imgur.com/4P4ZRxU.jpeg"

# Run the GPT-4 Vision model
response = api.run(task, img)

# Print the model's response
print(response)
```


### Text to Video with `ZeroscopeTTV`

```python
# Import the model
from swarms import ZeroscopeTTV

# Initialize the model
zeroscope = ZeroscopeTTV()

# Specify the task
task = "A person is walking on the street."

# Generate the video!
video_path = zeroscope(task)
print(video_path)

```


### Simple Conversational Agent
- Plug in and play conversational agent with `GPT4`, `Mixytral`, or any of our models
- Reliable conversational structure to hold messages together with dynamic handling for long context conversations and interactions with auto chunking
- Reliable, this simple system will always provide responses you want.
```python
import os

from dotenv import load_dotenv

from swarms import (
    OpenAIChat,
    Conversation,
)

conv = Conversation()

# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the language model
llm = OpenAIChat(openai_api_key=api_key, model_name="gpt-4")

# Run the language model in a loop
def interactive_conversation(llm):
    conv = Conversation()
    while True:
        user_input = input("User: ")
        conv.add("user", user_input)
        if user_input.lower() == "quit":
            break
        task = (
            conv.return_history_as_string()
        )  # Get the conversation history
        out = llm(task)
        conv.add("assistant", out)
        print(
            f"Assistant: {out}", #color="cyan"
        )
    conv.display_conversation()
    conv.export_conversation("conversation.txt")


# Replace with your LLM instance
interactive_conversation(llm)

```

---

# Features 🤖 
The Swarms framework is designed with a strong emphasis on reliability, performance, and production-grade readiness. 
Below are the key features that make Swarms an ideal choice for enterprise-level AI deployments.

## 🚀 Production-Grade Readiness
- **Scalable Architecture**: Built to scale effortlessly with your growing business needs.
- **Enterprise-Level Security**: Incorporates top-notch security features to safeguard your data and operations.
- **Containerization and Microservices**: Easily deployable in containerized environments, supporting microservices architecture.

## ⚙️ Reliability and Robustness
- **Fault Tolerance**: Designed to handle failures gracefully, ensuring uninterrupted operations.
- **Consistent Performance**: Maintains high performance even under heavy loads or complex computational demands.
- **Automated Backup and Recovery**: Features automatic backup and recovery processes, reducing the risk of data loss.

## 💡 Advanced AI Capabilities

The Swarms framework is equipped with a suite of advanced AI capabilities designed to cater to a wide range of applications and scenarios, ensuring versatility and cutting-edge performance.

### Multi-Modal Autonomous Agents
- **Versatile Model Support**: Seamlessly works with various AI models, including NLP, computer vision, and more, for comprehensive multi-modal capabilities.
- **Context-Aware Processing**: Employs context-aware processing techniques to ensure relevant and accurate responses from agents.

### Function Calling Models for API Execution
- **Automated API Interactions**: Function calling models that can autonomously execute API calls, enabling seamless integration with external services and data sources.
- **Dynamic Response Handling**: Capable of processing and adapting to responses from APIs for real-time decision making.

### Varied Architectures of Swarms
- **Flexible Configuration**: Supports multiple swarm architectures, from centralized to decentralized, for diverse application needs.
- **Customizable Agent Roles**: Allows customization of agent roles and behaviors within the swarm to optimize performance and efficiency.

### Generative Models
- **Advanced Generative Capabilities**: Incorporates state-of-the-art generative models to create content, simulate scenarios, or predict outcomes.
- **Creative Problem Solving**: Utilizes generative AI for innovative problem-solving approaches and idea generation.

### Enhanced Decision-Making
- **AI-Powered Decision Algorithms**: Employs advanced algorithms for swift and effective decision-making in complex scenarios.
- **Risk Assessment and Management**: Capable of assessing risks and managing uncertain situations with AI-driven insights.

### Real-Time Adaptation and Learning
- **Continuous Learning**: Agents can continuously learn and adapt from new data, improving their performance and accuracy over time.
- **Environment Adaptability**: Designed to adapt to different operational environments, enhancing robustness and reliability.


## 🔄 Efficient Workflow Automation
- **Streamlined Task Management**: Simplifies complex tasks with automated workflows, reducing manual intervention.
- **Customizable Workflows**: Offers customizable workflow options to fit specific business needs and requirements.
- **Real-Time Analytics and Reporting**: Provides real-time insights into agent performance and system health.

## 🌐 Wide-Ranging Integration
- **API-First Design**: Easily integrates with existing systems and third-party applications via robust APIs.
- **Cloud Compatibility**: Fully compatible with major cloud platforms for flexible deployment options.
- **Continuous Integration/Continuous Deployment (CI/CD)**: Supports CI/CD practices for seamless updates and deployment.

## 📊 Performance Optimization
- **Resource Management**: Efficiently manages computational resources for optimal performance.
- **Load Balancing**: Automatically balances workloads to maintain system stability and responsiveness.
- **Performance Monitoring Tools**: Includes comprehensive monitoring tools for tracking and optimizing performance.

## 🛡️ Security and Compliance
- **Data Encryption**: Implements end-to-end encryption for data at rest and in transit.
- **Compliance Standards Adherence**: Adheres to major compliance standards ensuring legal and ethical usage.
- **Regular Security Updates**: Regular updates to address emerging security threats and vulnerabilities.

## 💬 Community and Support
- **Extensive Documentation**: Detailed documentation for easy implementation and troubleshooting.
- **Active Developer Community**: A vibrant community for sharing ideas, solutions, and best practices.
- **Professional Support**: Access to professional support for enterprise-level assistance and guidance.

Swarms framework is not just a tool but a robust, scalable, and secure partner in your AI journey, ready to tackle the challenges of modern AI applications in a business environment.


## Documentation
- For documentation, go here, [swarms.apac.ai](https://swarms.apac.ai)


## 🫶 Contributions:

Swarms is an open-source project, and contributions are welcome. If you want to contribute, you can create new features, fix bugs, or improve the infrastructure. Please refer to the [CONTRIBUTING.md](https://github.com/kyegomez/swarms/blob/master/CONTRIBUTING.md) and our [contributing board](https://github.com/users/kyegomez/projects/1) file in the repository for more information on how to contribute.

To see how to contribute, visit [Contribution guidelines](https://github.com/kyegomez/swarms/blob/master/CONTRIBUTING.md)

<a href="https://github.com/kyegomez/swarms/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kyegomez/swarms" />
</a>


## Community
- [Join the Swarms community on Discord!](https://discord.gg/AJazBmhKnr)
- Join our Swarms Community Gathering every Thursday at 1pm NYC Time to unlock the potential of autonomous agents in automating your daily tasks [Sign up here](https://lu.ma/5p2jnc2v)



## Discovery Call
Book a discovery call with the Swarms team to learn how to optimize and scale your swarm! [Click here to book a time that works for you!](https://calendly.com/swarm-corp/30min?month=2023-11)

# License
Apache License

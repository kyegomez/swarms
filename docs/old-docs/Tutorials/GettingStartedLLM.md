# Getting Started with Swarms: A Simple Introduction to State-of-the-Art Language Models
======================================================================================

Welcome to the universe of Swarms! 🚀

Today, you're embarking on a thrilling journey through the ever-evolving realm of state-of-the-art language models.

As you might know, we're in the early days of this adventure, and every step we take is building from the ground up.

Our foundation is set on five levels of abstraction.

Each level adds complexity and capability, but worry not!

We'll walk you through each step, making sure you have fun and learn along the way.

So, ready to swarm?

Let's dive right in!

Installation 😊
===============

To get started with Swarms, run the following command:

pip install swarms

1\. OpenAI
==========

Ah, OpenAI, where the magic of GPT series lives.

With Swarms, you can tap into this magic in a straightforward way.

Think of it as having a chat with one of the smartest beings ever created by humankind!

Features ✨
----------

-   Direct Interface: Seamless interaction with OpenAI's GPT models.
-   Synchronous & Asynchronous Interaction: Flexibility to interact in real-time or in the background.
-   Multi-query Support: Enables querying multiple IDs simultaneously.
-   Streaming Capability: Stream multiple responses for dynamic conversations.
-   Console Logging: Gives users visibility and traceability of their interactions.

How It Works:
=============

1.  Initiate: Set up your agent using your OpenAI API key and other customizable parameters.
2.  Converse: Use methods like `generate` to converse with the model. Got a list of queries? No worries, methods like `ask_multiple` got you covered.
3.  Marvel: Witness the intelligence in the responses and interact in real-time!

Quick Start:
============

Imagine a scenario where you want to know how multiple IDs (say products, books, or places) are perceived. It's just two lines of code away!

from swarms import OpenAI()\
chat = OpenAI()\
response = chat.generate("Hello World")\
print(response)

2\. HuggingFace
===============

HuggingFace is a name that's changed the game in the NLP world. And with Swarms, you can easily harness the power of their vast model repository.

Features ✨
----------

-   Access to a Vast Model Repository: Directly tap into HuggingFace's expansive model hub.
-   Intuitive Text Generation: Prompt-based text generation that's straightforward.
-   High Customizability: Users can set device preferences, maximum length of generated text, and more.
-   Speed Boost: Our implementation offers up to a 9x speed increase by leveraging model quantization.
-   Less Memory Consumption: Quantization reduces the model size significantly.
-   Maintained Accuracy: Despite the reduction in model size and increased speed, the quality of the output remains top-tier.
-   Superior to Other Packages: Unlike many other packages that simply wrap around the HuggingFace API, Swarms has built-in support for advanced features like quantization, making it both faster and more efficient.

How It Works:
=============

1.  Pick Your Model: From BERT to GPT-2, choose from a myriad of options.
2.  Chat Away: Generate thought-provoking text based on your prompts.

Quick Start:
============

Ready to create a story?

from swarms import HuggingFaceLLM

hugging_face_model = HuggingFaceLLM(model_id="amazon/FalconLite")\
generated_text = hugging_face_model.generate("In a world where AI rules,"

3\. Google PaLM
===============

Google's venture into conversational AI, the PaLM Chat API, can now be effortlessly integrated into your projects with Swarms.

Features ✨
----------

-   Easy Integration: Quickly set up interactions with Google's PaLM Chat API.
-   Dynamic Conversations: Engage in back-and-forth chat-like conversations with the model.
-   Customizable Sampling Techniques: Set temperature, top-p, and top-k values for diverse and controlled outputs.

How It Works:
=============

1.  Set Up: Initialize with your preferred model and Google API key.
2.  Engage: Engage in back-and-forth conversations with the model.

Quick Start:
============

Looking for a quick joke? Google's got you:

from swarms import GooglePalm

google_palm = GooglePalm()\
messages = [{"role": "system", "content": "You are a funny assistant"}, {"role": "user", "content": "Crack me a joke"}]\
response = google_palm.generate(messages)

4\. Anthropic (swarms.models.Anthropic)
==============================================

Anthropic's models, with their mysterious allure, are now at your fingertips.

Features ✨
----------

-   Simplified Access: Straightforward interaction with Anthropic's large language models.
-   Dynamic Text Generation: Generate intriguing content based on user prompts.
-   Streaming Mode: Enable real-time streaming of responses for dynamic use-cases.

How It Works:
=============

1.  Initialize: Get started with your preferred Anthropic model.
2.  Generate: Whether you're crafting a story or looking for answers, you're in for a treat.

Quick Start:
============

Dive into a fairy tale:

from swarms import Anthropic

anthropic = Anthropic()\
generated_text = anthropic.generate("In a kingdom far away,")

Building with the Five Levels of Abstraction
============================================

From the individual model, right up to the hivemind, we've crafted a layered approach that scales and diversifies your interactions:

1.  Model: Start with a base model like OpenAI.
2.  Agent Level: Integrate the model with vector stores and tools.
3.  Worker Infrastructure: Assign tasks to worker nodes with specific tools.
4.  Swarm Level: Coordinate multiple worker nodes for a symphony of intelligence.
5.  Hivemind: The pinnacle! Integrate multiple swarms for unparalleled capability.

And, our master plan is...

The Master Plan
===============

Phase 1: Building the Foundation
--------------------------------

In the first phase, our focus is on building the basic infrastructure of Swarms.

This includes developing key components like the Swarms class, integrating essential tools, and establishing task completion and evaluation logic.

We'll also start developing our testing and evaluation framework during this phase.

If you're interested in foundational work and have a knack for building robust, scalable systems, this phase is for you.

Phase 2: Optimizing the System
------------------------------

In the second phase, we'll focus on optimizing Swarms by integrating more advanced features, improving the system's efficiency, and refining our testing and evaluation framework.

This phase involves more complex tasks, so if you enjoy tackling challenging problems and contributing to the development of innovative features, this is the phase for you.

Phase 3: Towards Super-Intelligence
-----------------------------------

The third phase of our bounty program is the most exciting --- this is where we aim to achieve super-intelligence.

In this phase, we'll be working on improving the swarm's capabilities, expanding its skills, and fine-tuning the system based on real-world testing and feedback.

If you're excited about the future of AI and want to contribute to a project that could potentially transform the digital world, this is the phase for you.

Remember, our roadmap is a guide, and we encourage you to bring your own ideas and creativity to the table.

We believe that every contribution, no matter how small, can make a difference.

So join us on this exciting journey and help us create the future of Swarms.

Hiring:
=======

We're hiring: Engineers, Researchers, Interns And, salesprofessionals to work on democratizing swarms, email me at with your story at `kye@apac.ai`

In Conclusion: A World of Possibilities
=======================================

There you have it!

A whirlwind tour through some of the most cutting-edge language models available today.

Remember, Swarms is like a treasure chest, and we're continually adding more jewels to it.

As Sir Jonathan Ive would say, "True simplicity is derived from so much more than just the absence of clutter and ornamentation, it's about bringing order to complexity."

Now, with the foundation of Swarms beneath your feet, you're well-equipped to soar to new heights.

So go on, experiment, explore, and have a blast!

The future of AI awaits you! 🌌🐝🎉

*Disclaimer: Remember, we're at the early stages, but every idea, every line of code, every interaction you have, is helping shape the future of Swarms. So, thank you for being a part of this exciting journey!*

Happy Swarming!


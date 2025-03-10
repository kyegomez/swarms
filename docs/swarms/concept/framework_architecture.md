# Swarms Framework Architecture


The Swarms package is designed to orchestrate and manage **swarms of agents**, enabling collaboration between multiple Large Language Models (LLMs) or other agent types to solve complex tasks. The architecture is modular and scalable, facilitating seamless integration of various agents, models, prompts, and tools. Below is an overview of the architectural components, along with instructions on where to find the corresponding documentation.



```
swarms/
├── agents/
├── artifacts/
├── cli/
├── memory/
├── models/ ---> Moved to swarm_models
├── prompts/
├── schemas/
├── structs/
├── telemetry/
├── tools/
├── utils/
└── __init__.py
```



### Role of Folders in the Swarms Framework

The **Swarms framework** is composed of several key folders, each serving a specific role in building, orchestrating, and managing swarms of agents. Below is an in-depth explanation of the role of each folder in the framework's architecture, focusing on how they contribute to the overall system for handling complex multi-agent workflows.

---

### **1. Agents Folder (`agents/`)**
   - **Role:** 
     - The **agents** folder contains the core logic for individual agents within the Swarms framework. Agents are the key functional units responsible for carrying out specific tasks, whether it be text generation, web scraping, data analysis, or more specialized functions like marketing or accounting.
     - **Customization:** Each agent can be specialized for different tasks by defining custom system prompts and behaviors.
     - **Modular Agent System:** New agents can be easily added to this folder to expand the framework's capabilities.
   - **Importance:** This folder allows users to create and manage multiple types of agents that can interact and collaborate to solve complex problems.
   - **Examples:** Accounting agents, marketing agents, and programming agents.

---

### **2. Artifacts Folder (`artifacts/`)**
   - **Role:** 
     - The **artifacts** folder is responsible for storing the results or outputs generated by agents and swarms. This could include reports, logs, or data that agents generate during task execution.
     - **Persistent Storage:** It helps maintain a persistent record of agent interactions, making it easier to retrieve or review past actions and outputs.
     - **Data Handling:** Users can configure this folder to store artifacts that are essential for later analysis or reporting.
   - **Importance:** Acts as a storage mechanism for important task-related outputs, ensuring that no data is lost after tasks are completed.

---

### **3. CLI Folder (`cli/`)**
   - **Role:** 
     - The **CLI** folder contains tools for interacting with the Swarms framework through the command-line interface. This allows users to easily manage and orchestrate swarms without needing a graphical interface.
     - **Command-line Tools:** Commands in this folder enable users to initiate, control, and monitor swarms, making the system accessible and versatile.
     - **Automation and Scriptability:** Enables advanced users to automate swarm interactions and deploy agents programmatically.
   - **Importance:** Provides a flexible way to control the Swarms system for developers who prefer using the command line.

---

### **4. Memory Folder (`memory/`) Deprecated!!**
   - **Role:** 
     - The **memory** folder handles the framework's memory management for agents. This allows agents to retain and recall past interactions or task contexts, enabling continuity in long-running processes or multi-step workflows.
     - **Context Retention:** Agents that depend on historical context to make decisions or carry out tasks can store and access memory using this folder.
     - **Long-Term and Short-Term Memory:** This could be implemented in various ways, such as short-term conversational memory or long-term knowledge storage.
   - **Importance:** Crucial for agents that require memory to handle complex workflows, where decisions are based on prior outputs or interactions.

---

### **5. Models Folder (`models/`) Moved to `swarm_models`**
   - **Role:** 
     - The **models** folder houses pre-trained machine learning models that agents utilize to complete their tasks. These models could include LLMs (Large Language Models), custom-trained models, or fine-tuned models specific to the tasks being handled by the agents.
     - **Plug-and-Play Architecture:** The framework allows users to easily add or switch models depending on the specific needs of their agents.
     - **Custom Model Support:** Users can integrate custom models here for more specialized tasks.
   - **Importance:** Provides the computational backbone for agent decision-making and task execution.

---

### **6. Prompts Folder (`prompts/`)**
   - **Role:** 
     - The **prompts** folder contains reusable prompt templates that agents use to interact with their environment and complete tasks. These system prompts define the behavior and task orientation of the agents.
     - **Template Reusability:** Users can create and store common prompt templates, making it easy to define agent behavior across different tasks without rewriting prompts from scratch.
     - **Task-Specific Prompts:** For example, an accounting agent may have a prompt template that guides its interaction with financial data.
   - **Importance:** Provides the logic and guidance agents need to generate outputs in a coherent and task-focused manner.

---

### **7. Schemas Folder (`schemas/`)**
   - **Role:** 
     - The **schemas** folder defines the data structures and validation logic for inputs and outputs within the framework, using tools like **Pydantic** for data validation.
     - **Standardization and Validation:** This ensures that all interactions between agents and swarms follow consistent data formats, which is critical for large-scale agent coordination and task management.
     - **Error Prevention:** By validating data early, it prevents errors from propagating through the system, improving reliability.
   - **Importance:** Ensures data consistency across the entire framework, making it easier to integrate and manage swarms of agents at scale.

---

### **8. Structs Folder (`structs/`)**
   - **Role:** 
     - The **structs** folder is the core of the Swarms framework, housing the orchestration logic for managing and coordinating swarms of agents. This folder allows for dynamic task assignment, queue management, inter-agent communication, and result aggregation.
     - **Swarm Management:** Agents are grouped into swarms to handle tasks that require multiple agents working in parallel or collaboratively.
     - **Scalability:** The swarm structure is designed to be scalable, allowing thousands of agents to operate together on distributed tasks.
     - **Task Queueing and Execution:** Supports task queueing, task prioritization, and load balancing between agents.
   - **Importance:** This folder is critical for managing how agents interact and collaborate to solve complex, multi-step problems.

---

### **9. Telemetry Folder (`telemetry/`)**
   - **Role:** 
     - The **telemetry** folder provides logging and monitoring tools to capture agent performance metrics, error handling, and real-time activity tracking. It helps users keep track of what each agent or swarm is doing, making it easier to debug, audit, and optimize operations.
     - **Monitoring:** Tracks agent performance and system health.
     - **Logs:** Maintains logs for troubleshooting and operational review.
   - **Importance:** Provides visibility into the system, ensuring smooth operation and enabling fine-tuning of agent behaviors.

---

### **10. Tools Folder (`tools/`)**
   - **Role:** 
     - The **tools** folder contains specialized utility functions or scripts that agents and swarms may require to complete certain tasks, such as web scraping, API interactions, data parsing, or other external resource handling.
     - **Task-Specific Tools:** Agents can call these tools to perform operations outside of their own logic, enabling them to interact with external systems more efficiently.
   - **Importance:** Expands the capabilities of agents, allowing them to complete more sophisticated tasks by relying on these external tools.

---

### **11. Utils Folder (`utils/`)**
   - **Role:** 
     - The **utils** folder contains general-purpose utility functions that are reused throughout the framework. These may include functions for data formatting, validation, logging setup, and configuration management.
     - **Shared Utilities:** Helps keep the codebase clean by providing reusable functions that multiple agents or parts of the framework can call.
   - **Importance:** Provides common functions that help the Swarms framework operate efficiently and consistently.

---

### **Core Initialization File (`__init__.py`)**
   - **Role:** 
     - The `__init__.py` file is the entry point of the Swarms package, ensuring that all necessary modules, agents, and tools are loaded when the Swarms framework is imported. It allows for the modular loading of different components, making it easier for users to work with only the parts of the framework they need.
   - **Importance:** Acts as the bridge that connects all other components in the framework, enabling the entire package to work together seamlessly.

---

### How to Access Documentation

- **Official Documentation Site:**
  - URL: [docs.swarms.world](https://docs.swarms.world)
  - Here, users can find detailed guides, tutorials, and API references on how to use each of the folders mentioned above. The documentation covers setup, agent orchestration, and practical examples of how to leverage swarms for real-world tasks.

- **GitHub Repository:**
  - URL: [Swarms GitHub](https://github.com/kyegomez/swarms)
  - The repository contains code examples, detailed folder explanations, and further resources on how to get started with building and managing agent swarms.

By understanding the purpose and role of each folder in the Swarms framework, users can more effectively build, orchestrate, and manage agents to handle complex tasks and workflows at scale.

## Support:

- **Post Issue On Github**
  - URL: [Submit issue](https://github.com/kyegomez/swarms/issues/new/choose)
  - Post your issue whether it's an issue or a feature request


- **Community Support**
  - URL: [Submit issue](https://discord.gg/jM3Z6M9uMq)
  - Ask the community for support in real-time and or admin support
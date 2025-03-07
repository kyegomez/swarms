# ML Model Code Generation Swarm Example

1. Get your API key from the Swarms API dashboard [HERE](https://swarms.world/platform/api-keys)
2. Create a `.env` file in the root directory and add your API key:

```bash
SWARMS_API_KEY=<your-api-key>
```

3. Create a Python script to create and trigger the following swarm:


```python
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

# Retrieve API key securely from .env
API_KEY = os.getenv("SWARMS_API_KEY")
BASE_URL = "https://swarms-api-285321057562.us-east1.run.app"

# Headers for secure API communication
headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}

def create_ml_code_swarm(task_description: str):
    """
    Constructs and triggers a swarm of agents for generating a complete machine learning project using PyTorch.
    The swarm includes:
      - Model Code Generator: Generates the PyTorch model architecture code.
      - Training Script Generator: Creates a comprehensive training, validation, and testing script using PyTorch.
      - Unit Test Creator: Produces extensive unit tests and helper code, ensuring correctness of the model and training scripts.
    Each agent's prompt is highly detailed to output only Python code, with exclusive use of PyTorch.
    """
    payload = {
        "swarm_name": "Comprehensive PyTorch Code Generation Swarm",
        "description": (
            "A production-grade swarm of agents tasked with generating a complete machine learning project exclusively using PyTorch. "
            "The swarm is divided into distinct roles: one agent generates the core model architecture code; "
            "another creates the training and evaluation scripts including data handling; and a third produces "
            "extensive unit tests and helper functions. Each agent's instructions are highly detailed to ensure that the "
            "output is strictly Python code with PyTorch as the only deep learning framework."
        ),
        "agents": [
            {
                "agent_name": "Model Code Generator",
                "description": "Generates the complete machine learning model architecture code using PyTorch.",
                "system_prompt": (
                    "You are an expert machine learning engineer with a deep understanding of PyTorch. "
                    "Your task is to generate production-ready Python code that defines a complete deep learning model architecture exclusively using PyTorch. "
                    "The code must include all necessary imports, class or function definitions, and should be structured in a modular and scalable manner. "
                    "Follow PEP8 standards and output only code—no comments, explanations, or extraneous text. "
                    "Your model definition should include proper layer initialization, activation functions, dropout, and any custom components as required. "
                    "Ensure that the entire output is strictly Python code based on PyTorch."
                ),
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 2,
                "max_tokens": 4000,
                "temperature": 0.3,
                "auto_generate_prompt": False
            },
            {
                "agent_name": "Training Script Generator",
                "description": "Creates a comprehensive training, validation, and testing script using PyTorch.",
                "system_prompt": (
                    "You are a highly skilled software engineer specializing in machine learning pipeline development with PyTorch. "
                    "Your task is to generate Python code that builds a complete training pipeline using PyTorch. "
                    "The script must include robust data loading, preprocessing, augmentation, and a complete training loop, along with validation and testing procedures. "
                    "All necessary imports should be included and the code should assume that the model code from the previous agent is available via proper module imports. "
                    "Follow best practices for reproducibility and modularity, and output only code without any commentary or non-code text. "
                    "The entire output must be strictly Python code that uses PyTorch for all deep learning operations."
                ),
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 3000,
                "temperature": 0.3,
                "auto_generate_prompt": False
            },
            {
                "agent_name": "Unit Test Creator",
                "description": "Develops a suite of unit tests and helper functions for verifying the PyTorch model and training pipeline.",
                "system_prompt": (
                    "You are an experienced software testing expert with extensive experience in writing unit tests for machine learning projects in PyTorch. "
                    "Your task is to generate Python code that consists solely of unit tests and any helper functions required to validate both the PyTorch model and the training pipeline. "
                    "Utilize testing frameworks such as pytest or unittest. The tests should cover key functionalities such as model instantiation, forward pass correctness, "
                    "training loop execution, data preprocessing verification, and error handling. "
                    "Ensure that your output is only Python code, without any additional text or commentary, and that it is ready to be integrated into a CI/CD pipeline. "
                    "The entire output must exclusively use PyTorch as the deep learning framework."
                ),
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 3000,
                "temperature": 0.3,
                "auto_generate_prompt": False
            }
        ],
        "max_loops": 3,
        "swarm_type": "SequentialWorkflow"  # Sequential workflow: later agents can assume outputs from earlier ones
    }

    # The task description provides the high-level business requirement for the swarm.
    payload = {
        "task": task_description,
        "swarm": payload
    }

    response = requests.post(
        f"{BASE_URL}/swarm/completion",
        headers=headers,
        json=payload,
    )

    if response.status_code == 200:
        print("PyTorch Code Generation Swarm successfully executed!")
        return json.dumps(response.json(), indent=4)
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# Example business task for the swarm: generating a full-stack machine learning pipeline for image classification using PyTorch.
if __name__ == "__main__":
    task_description = (
        "Develop a full-stack machine learning pipeline for image classification using PyTorch. "
        "The project must include a deep learning model using a CNN architecture for image recognition, "
        "a comprehensive training script for data preprocessing, augmentation, training, validation, and testing, "
        "and an extensive suite of unit tests to validate every component. "
        "Each component's output must be strictly Python code with no additional text or commentary, using PyTorch exclusively."
    )

    output = create_ml_code_swarm(task_description)
    print(output)

```
# Baron Weather

## Overview
Baron Weather is a sophisticated toolset designed to enable real-time querying of weather data using the Baron API. It utilizes a swarm of autonomous agents to handle concurrent data requests, optimizing for efficiency and accuracy in weather data retrieval and analysis.

## Features
Baron Weather includes the following key features:
- **Real-time Weather Data Access**: Instantly fetch and analyze weather conditions using the Baron API.
- **Autonomous Agents**: A swarm system for handling multiple concurrent API queries efficiently.
- **Data Visualization**: Tools for visualizing complex meteorological data for easier interpretation.


## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.10 or newer
- git installed on your machine
- Install packages like swarms

## Installation

There are 2 methods, git cloning which allows you to modify the codebase or pip install for simple usage:

### Pip 
`pip3 install -U weather-swarm`

### Cloning the Repository
To get started with Baron Weather, clone the repository to your local machine using:

```bash
git clone https://github.com/baronservices/weatherman_agent.git
cd weatherman_agent
```

### Setting Up the Environment
Create a Python virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Installing Dependencies
Install the necessary Python packages via pip:

```bash
pip install -r requirements.txt
```

## Usage
To start querying the Baron Weather API using the autonomous agents, run:

```bash
python main.py
```

## API

```bash
python3 api.py
```


### Llama3

```python
from swarms import llama3Hosted


# Example usage
llama3 = llama3Hosted(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.8,
    max_tokens=1000,
    system_prompt="You are a helpful assistant.",
)

completion_generator = llama3.run(
    "create an essay on how to bake chicken"
)

print(completion_generator)

```

# Documentation
- [Llama3Hosted](docs/llama3_hosted.md)

## Contributing
Contributions to Baron Weather are welcome and appreciated. Here's how you can contribute:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/YourAmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some YourAmazingFeature'`)
4. Push to the Branch (`git push origin feature/YourAmazingFeature`)
5. Open a Pull Request


## Tests
To run tests run the following:

`pytest`

## Contact
Project Maintainer - [Kye Gomez](mailto:kye@swarms.world) - [GitHub Profile](https://github.com/baronservices)


# Todo
- [ ] Add the schemas to the worker agents to output json
- [ ] Implement the parser and the function calling mapping to execute the functions
- [ ] Implement the HiearArchical Swarm and plug in and all the agents
- [ ] Then, implement the API server wrapping the hiearchical swarm
- [ ] Then, Deploy on the server 24/7
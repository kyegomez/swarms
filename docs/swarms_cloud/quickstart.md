
# Swarms Quickstart Guide

This guide will help you get started with both single agent and multi-agent functionalities in Swarms API.

## Prerequisites

!!! info "Requirements"

    - Python 3.7+
    - API key from [Swarms Platform](https://swarms.world/platform/api-keys)
    - `requests` library for Python
    - `axios` for TypeScript/JavaScript
    - `curl` for shell commands

## Installation

=== "pip"

    ```bash
    pip install requests python-dotenv
    ```

=== "npm"

    ```bash
    npm install axios dotenv
    ```

## Authentication

!!! warning "API Key Security"

    Never hardcode your API key in your code. Always use environment variables or secure configuration management.

The API is accessible through two base URLs:

- Production: `https://api.swarms.world`
- Alternative: `https://swarms-api-285321057562.us-east1.run.app`

## Single Agent Usage

### Health Check

=== "Python"

    ```python linenums="1" title="health_check.py"
    import os
    import requests
    from dotenv import load_dotenv

    load_dotenv()
    API_KEY = os.getenv("SWARMS_API_KEY")
    BASE_URL = "https://api.swarms.world"

    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.get(f"{BASE_URL}/health", headers=headers)
    print(response.json())
    ```

=== "cURL"

    ```bash title="health_check.sh"
    curl -X GET "https://api.swarms.world/health" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json"
    ```

=== "TypeScript"

    ```typescript linenums="1" title="health_check.ts"
    import axios from 'axios';
    import * as dotenv from 'dotenv';

    dotenv.config();
    const API_KEY = process.env.SWARMS_API_KEY;
    const BASE_URL = 'https://api.swarms.world';

    async function checkHealth() {
      try {
        const response = await axios.get(`${BASE_URL}/health`, {
          headers: {
            'x-api-key': API_KEY,
            'Content-Type': 'application/json'
          }
        });
        console.log(response.data);
      } catch (error) {
        console.error('Error:', error);
      }
    }

    checkHealth();
    ```

### Basic Agent

=== "Python"

    ```python linenums="1" title="single_agent.py"
    import os
    import requests
    from dotenv import load_dotenv

    load_dotenv()

    API_KEY = os.getenv("SWARMS_API_KEY")  # (1)
    BASE_URL = "https://api.swarms.world"

    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    def run_single_agent():
        """Run a single agent with the AgentCompletion format"""
        payload = {
            "agent_config": {
                "agent_name": "Research Analyst",  # (2)
                "description": "An expert in analyzing and synthesizing research data",
                "system_prompt": (  # (3)
                    "You are a Research Analyst with expertise in data analysis and synthesis. "
                    "Your role is to analyze provided information, identify key insights, "
                    "and present findings in a clear, structured format."
                ),
                "model_name": "claude-3-5-sonnet-20240620",  # (4)
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 1,
                "auto_generate_prompt": False,
                "tools_list_dictionary": None,
            },
            "task": "What are the key trends in renewable energy adoption?",  # (5)
        }

        response = requests.post(
            f"{BASE_URL}/v1/agent/completions",
            headers=headers,
            json=payload
        )
        return response.json()

    # Run the agent
    result = run_single_agent()
    print(result)
    ```

    1. Load API key from environment variables
    2. Give your agent a descriptive name
    3. Define the agent's capabilities and role
    4. Choose from available models
    5. Specify the task for the agent

=== "cURL"

    ```bash title="single_agent.sh"
    curl -X POST "https://api.swarms.world/v1/agent/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "agent_config": {
          "agent_name": "Research Analyst",
          "description": "An expert in analyzing and synthesizing research data",
          "system_prompt": "You are a Research Analyst with expertise in data analysis and synthesis. Your role is to analyze provided information, identify key insights, and present findings in a clear, structured format.",
          "model_name": "claude-3-5-sonnet-20240620",
          "role": "worker",
          "max_loops": 1,
          "max_tokens": 8192,
          "temperature": 1,
          "auto_generate_prompt": false,
          "tools_list_dictionary": null
        },
        "task": "What are the key trends in renewable energy adoption?"
      }'
    ```

=== "TypeScript"

    ```typescript linenums="1" title="single_agent.ts"
    import axios from 'axios';
    import * as dotenv from 'dotenv';

    dotenv.config();

    const API_KEY = process.env.SWARMS_API_KEY;
    const BASE_URL = 'https://api.swarms.world';

    interface AgentConfig {
      agent_name: string;
      description: string;
      system_prompt: string;
      model_name: string;
      role: string;
      max_loops: number;
      max_tokens: number;
      temperature: number;
      auto_generate_prompt: boolean;
      tools_list_dictionary: null | object[];
    }

    interface AgentPayload {
      agent_config: AgentConfig;
      task: string;
    }

    async function runSingleAgent() {
      const payload: AgentPayload = {
        agent_config: {
          agent_name: "Research Analyst",
          description: "An expert in analyzing and synthesizing research data",
          system_prompt: "You are a Research Analyst with expertise in data analysis and synthesis.",
          model_name: "claude-3-5-sonnet-20240620",
          role: "worker",
          max_loops: 1,
          max_tokens: 8192,
          temperature: 1,
          auto_generate_prompt: false,
          tools_list_dictionary: null
        },
        task: "What are the key trends in renewable energy adoption?"
      };

      try {
        const response = await axios.post(
          `${BASE_URL}/v1/agent/completions`,
          payload,
          {
            headers: {
              'x-api-key': API_KEY,
              'Content-Type': 'application/json'
            }
          }
        );
        return response.data;
      } catch (error) {
        console.error('Error:', error);
        throw error;
      }
    }

    // Run the agent
    runSingleAgent()
      .then(result => console.log(result))
      .catch(error => console.error(error));
    ```

### Agent with History

=== "Python"

    ```python linenums="1" title="agent_with_history.py"
    def run_agent_with_history():
        payload = {
            "agent_config": {
                "agent_name": "Conversation Agent",
                "description": "An agent that maintains conversation context",
                "system_prompt": "You are a helpful assistant that maintains context.",
                "model_name": "claude-3-5-sonnet-20240620",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.7,
                "auto_generate_prompt": False,
            },
            "task": "What's the weather like?",
            "history": [  # (1)
                {
                    "role": "user",
                    "content": "I'm planning a trip to New York."
                },
                {
                    "role": "assistant",
                    "content": "That's great! When are you planning to visit?"
                },
                {
                    "role": "user",
                    "content": "Next week."
                }
            ]
        }

        response = requests.post(
            f"{BASE_URL}/v1/agent/completions",
            headers=headers,
            json=payload
        )
        return response.json()
    ```

    1. Include conversation history for context

=== "cURL"

    ```bash title="agent_with_history.sh"
    curl -X POST "https://api.swarms.world/v1/agent/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "agent_config": {
          "agent_name": "Conversation Agent",
          "description": "An agent that maintains conversation context",
          "system_prompt": "You are a helpful assistant that maintains context.",
          "model_name": "claude-3-5-sonnet-20240620",
          "role": "worker",
          "max_loops": 1,
          "max_tokens": 8192,
          "temperature": 0.7,
          "auto_generate_prompt": false
        },
        "task": "What'\''s the weather like?",
        "history": [
          {
            "role": "user",
            "content": "I'\''m planning a trip to New York."
          },
          {
            "role": "assistant",
            "content": "That'\''s great! When are you planning to visit?"
          },
          {
            "role": "user",
            "content": "Next week."
          }
        ]
      }'
    ```

=== "TypeScript"

    ```typescript linenums="1" title="agent_with_history.ts"
    interface Message {
      role: 'user' | 'assistant';
      content: string;
    }

    interface AgentWithHistoryPayload extends AgentPayload {
      history: Message[];
    }

    async function runAgentWithHistory() {
      const payload: AgentWithHistoryPayload = {
        agent_config: {
          agent_name: "Conversation Agent",
          description: "An agent that maintains conversation context",
          system_prompt: "You are a helpful assistant that maintains context.",
          model_name: "claude-3-5-sonnet-20240620",
          role: "worker",
          max_loops: 1,
          max_tokens: 8192,
          temperature: 0.7,
          auto_generate_prompt: false,
          tools_list_dictionary: null
        },
        task: "What's the weather like?",
        history: [
          {
            role: "user",
            content: "I'm planning a trip to New York."
          },
          {
            role: "assistant",
            content: "That's great! When are you planning to visit?"
          },
          {
            role: "user",
            content: "Next week."
          }
        ]
      };

      try {
        const response = await axios.post(
          `${BASE_URL}/v1/agent/completions`,
          payload,
          {
            headers: {
              'x-api-key': API_KEY,
              'Content-Type': 'application/json'
            }
          }
        );
        return response.data;
      } catch (error) {
        console.error('Error:', error);
        throw error;
      }
    }
    ```

## Multi-Agent Swarms

!!! tip "Swarm Types"

    Swarms API supports two types of agent workflows:
    
    1. `SequentialWorkflow`: Agents work in sequence, each building on previous output
    2. `ConcurrentWorkflow`: Agents work in parallel on the same task

### Sequential Workflow

=== "Python"

    ```python linenums="1" title="sequential_swarm.py"
    def run_sequential_swarm():
        payload = {
            "name": "Financial Analysis Swarm",
            "description": "Market analysis swarm",
            "agents": [
                {
                    "agent_name": "Market Analyst",  # (1)
                    "description": "Analyzes market trends",
                    "system_prompt": "You are a financial analyst expert.",
                    "model_name": "gpt-4o",
                    "role": "worker",
                    "max_loops": 1,
                    "max_tokens": 8192,
                    "temperature": 0.5,
                    "auto_generate_prompt": False
                },
                {
                    "agent_name": "Economic Forecaster",  # (2)
                    "description": "Predicts economic trends",
                    "system_prompt": "You are an expert in economic forecasting.",
                    "model_name": "gpt-4o",
                    "role": "worker",
                    "max_loops": 1,
                    "max_tokens": 8192,
                    "temperature": 0.5,
                    "auto_generate_prompt": False
                }
            ],
            "max_loops": 1,
            "swarm_type": "SequentialWorkflow",  # (3)
            "task": "Analyze the current market conditions and provide economic forecasts."
        }

        response = requests.post(
            f"{BASE_URL}/v1/swarm/completions",
            headers=headers,
            json=payload
        )
        return response.json()
    ```

    1. First agent analyzes market trends
    2. Second agent builds on first agent's analysis
    3. Sequential workflow ensures ordered execution

=== "cURL"

    ```bash title="sequential_swarm.sh"
    curl -X POST "https://api.swarms.world/v1/swarm/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Financial Analysis Swarm",
        "description": "Market analysis swarm",
        "agents": [
          {
            "agent_name": "Market Analyst",
            "description": "Analyzes market trends",
            "system_prompt": "You are a financial analyst expert.",
            "model_name": "gpt-4o",
            "role": "worker",
            "max_loops": 1,
            "max_tokens": 8192,
            "temperature": 0.5,
            "auto_generate_prompt": false
          },
          {
            "agent_name": "Economic Forecaster",
            "description": "Predicts economic trends",
            "system_prompt": "You are an expert in economic forecasting.",
            "model_name": "gpt-4o",
            "role": "worker",
            "max_loops": 1,
            "max_tokens": 8192,
            "temperature": 0.5,
            "auto_generate_prompt": false
          }
        ],
        "max_loops": 1,
        "swarm_type": "SequentialWorkflow",
        "task": "Analyze the current market conditions and provide economic forecasts."
      }'
    ```

=== "TypeScript"

    ```typescript linenums="1" title="sequential_swarm.ts"
    interface SwarmAgent {
      agent_name: string;
      description: string;
      system_prompt: string;
      model_name: string;
      role: string;
      max_loops: number;
      max_tokens: number;
      temperature: number;
      auto_generate_prompt: boolean;
    }

    interface SwarmPayload {
      name: string;
      description: string;
      agents: SwarmAgent[];
      max_loops: number;
      swarm_type: 'SequentialWorkflow' | 'ConcurrentWorkflow';
      task: string;
    }

    async function runSequentialSwarm() {
      const payload: SwarmPayload = {
        name: "Financial Analysis Swarm",
        description: "Market analysis swarm",
        agents: [
          {
            agent_name: "Market Analyst",
            description: "Analyzes market trends",
            system_prompt: "You are a financial analyst expert.",
            model_name: "gpt-4o",
            role: "worker",
            max_loops: 1,
            max_tokens: 8192,
            temperature: 0.5,
            auto_generate_prompt: false
          },
          {
            agent_name: "Economic Forecaster",
            description: "Predicts economic trends",
            system_prompt: "You are an expert in economic forecasting.",
            model_name: "gpt-4o",
            role: "worker",
            max_loops: 1,
            max_tokens: 8192,
            temperature: 0.5,
            auto_generate_prompt: false
          }
        ],
        max_loops: 1,
        swarm_type: "SequentialWorkflow",
        task: "Analyze the current market conditions and provide economic forecasts."
      };

      try {
        const response = await axios.post(
          `${BASE_URL}/v1/swarm/completions`,
          payload,
          {
            headers: {
              'x-api-key': API_KEY,
              'Content-Type': 'application/json'
            }
          }
        );
        return response.data;
      } catch (error) {
        console.error('Error:', error);
        throw error;
      }
    }
    ```

### Concurrent Workflow

=== "Python"

    ```python linenums="1" title="concurrent_swarm.py"
    def run_concurrent_swarm():
        payload = {
            "name": "Medical Analysis Swarm",
            "description": "Analyzes medical data concurrently",
            "agents": [
                {
                    "agent_name": "Lab Data Analyzer",  # (1)
                    "description": "Analyzes lab report data",
                    "system_prompt": "You are a medical data analyst specializing in lab results.",
                    "model_name": "claude-3-5-sonnet-20240620",
                    "role": "worker",
                    "max_loops": 1,
                    "max_tokens": 8192,
                    "temperature": 0.5,
                    "auto_generate_prompt": False
                },
                {
                    "agent_name": "Clinical Specialist",  # (2)
                    "description": "Provides clinical interpretations",
                    "system_prompt": "You are an expert in clinical diagnosis.",
                    "model_name": "claude-3-5-sonnet-20240620",
                    "role": "worker",
                    "max_loops": 1,
                    "max_tokens": 8192,
                    "temperature": 0.5,
                    "auto_generate_prompt": False
                }
            ],
            "max_loops": 1,
            "swarm_type": "ConcurrentWorkflow",  # (3)
            "task": "Analyze these lab results and provide clinical interpretations."
        }

        response = requests.post(
            f"{BASE_URL}/v1/swarm/completions",
            headers=headers,
            json=payload
        )
        return response.json()
    ```

    1. First agent processes lab data
    2. Second agent works simultaneously
    3. Concurrent workflow for parallel processing

=== "cURL"

    ```bash title="concurrent_swarm.sh"
    curl -X POST "https://api.swarms.world/v1/swarm/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Medical Analysis Swarm",
        "description": "Analyzes medical data concurrently",
        "agents": [
          {
            "agent_name": "Lab Data Analyzer",
            "description": "Analyzes lab report data",
            "system_prompt": "You are a medical data analyst specializing in lab results.",
            "model_name": "claude-3-5-sonnet-20240620",
            "role": "worker",
            "max_loops": 1,
            "max_tokens": 8192,
            "temperature": 0.5,
            "auto_generate_prompt": false
          },
          {
            "agent_name": "Clinical Specialist",
            "description": "Provides clinical interpretations",
            "system_prompt": "You are an expert in clinical diagnosis.",
            "model_name": "claude-3-5-sonnet-20240620",
            "role": "worker",
            "max_loops": 1,
            "max_tokens": 8192,
            "temperature": 0.5,
            "auto_generate_prompt": false
          }
        ],
        "max_loops": 1,
        "swarm_type": "ConcurrentWorkflow",
        "task": "Analyze these lab results and provide clinical interpretations."
      }'
    ```

=== "TypeScript"

    ```typescript linenums="1" title="concurrent_swarm.ts"
    async function runConcurrentSwarm() {
      const payload: SwarmPayload = {
        name: "Medical Analysis Swarm",
        description: "Analyzes medical data concurrently",
        agents: [
          {
            agent_name: "Lab Data Analyzer",
            description: "Analyzes lab report data",
            system_prompt: "You are a medical data analyst specializing in lab results.",
            model_name: "claude-3-5-sonnet-20240620",
            role: "worker",
            max_loops: 1,
            max_tokens: 8192,
            temperature: 0.5,
            auto_generate_prompt: false
          },
          {
            agent_name: "Clinical Specialist",
            description: "Provides clinical interpretations",
            system_prompt: "You are an expert in clinical diagnosis.",
            model_name: "claude-3-5-sonnet-20240620",
            role: "worker",
            max_loops: 1,
            max_tokens: 8192,
            temperature: 0.5,
            auto_generate_prompt: false
          }
        ],
        max_loops: 1,
        swarm_type: "ConcurrentWorkflow",
        task: "Analyze these lab results and provide clinical interpretations."
      };

      try {
        const response = await axios.post(
          `${BASE_URL}/v1/swarm/completions`,
          payload,
          {
            headers: {
              'x-api-key': API_KEY,
              'Content-Type': 'application/json'
            }
          }
        );
        return response.data;
      } catch (error) {
        console.error('Error:', error);
        throw error;
      }
    }
    ```

### Batch Processing

!!! example "Batch Processing"

    Process multiple swarms in a single request for improved efficiency.

=== "Python"

    ```python linenums="1" title="batch_swarms.py"
    def run_batch_swarms():
        payload = [
            {
                "name": "Batch Swarm 1",
                "description": "First swarm in batch",
                "agents": [
                    {
                        "agent_name": "Research Agent",
                        "description": "Conducts research",
                        "system_prompt": "You are a research assistant.",
                        "model_name": "gpt-4",
                        "role": "worker",
                        "max_loops": 1
                    },
                    {
                        "agent_name": "Analysis Agent",
                        "description": "Analyzes data",
                        "system_prompt": "You are a data analyst.",
                        "model_name": "gpt-4",
                        "role": "worker",
                        "max_loops": 1
                    }
                ],
                "max_loops": 1,
                "swarm_type": "SequentialWorkflow",
                "task": "Research AI advancements."
            }
        ]

        response = requests.post(
            f"{BASE_URL}/v1/swarm/batch/completions",
            headers=headers,
            json=payload
        )
        return response.json()
    ```

=== "cURL"

    ```bash title="batch_swarms.sh"
    curl -X POST "https://api.swarms.world/v1/swarm/batch/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '[
        {
          "name": "Batch Swarm 1",
          "description": "First swarm in batch",
          "agents": [
            {
              "agent_name": "Research Agent",
              "description": "Conducts research",
              "system_prompt": "You are a research assistant.",
              "model_name": "gpt-4",
              "role": "worker",
              "max_loops": 1
            },
            {
              "agent_name": "Analysis Agent",
              "description": "Analyzes data",
              "system_prompt": "You are a data analyst.",
              "model_name": "gpt-4",
              "role": "worker",
              "max_loops": 1
            }
          ],
          "max_loops": 1,
          "swarm_type": "SequentialWorkflow",
          "task": "Research AI advancements."
        }
      ]'
    ```

=== "TypeScript"

    ```typescript linenums="1" title="batch_swarms.ts"
    async function runBatchSwarms() {
      const payload: SwarmPayload[] = [
        {
          name: "Batch Swarm 1",
          description: "First swarm in batch",
          agents: [
            {
              agent_name: "Research Agent",
              description: "Conducts research",
              system_prompt: "You are a research assistant.",
              model_name: "gpt-4",
              role: "worker",
              max_loops: 1,
              max_tokens: 8192,
              temperature: 0.7,
              auto_generate_prompt: false
            },
            {
              agent_name: "Analysis Agent",
              description: "Analyzes data",
              system_prompt: "You are a data analyst.",
              model_name: "gpt-4",
              role: "worker",
              max_loops: 1,
              max_tokens: 8192,
              temperature: 0.7,
              auto_generate_prompt: false
            }
          ],
          max_loops: 1,
          swarm_type: "SequentialWorkflow",
          task: "Research AI advancements."
        }
      ];

      try {
        const response = await axios.post(
          `${BASE_URL}/v1/swarm/batch/completions`,
          payload,
          {
            headers: {
              'x-api-key': API_KEY,
              'Content-Type': 'application/json'
            }
          }
        );
        return response.data;
      } catch (error) {
        console.error('Error:', error);
        throw error;
      }
    }
    ```

## Advanced Features

### Tools Integration

!!! note "Tools"

    Enhance agent capabilities by providing them with specialized tools.

=== "Python"

    ```python linenums="1" title="tools_example.py"
    def run_agent_with_tools():
        tools_dictionary = [
            {
                "type": "function",
                "function": {
                    "name": "search_topic",
                    "description": "Conduct an in-depth search on a topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "depth": {
                                "type": "integer",
                                "description": "Search depth (1-3)"
                            },
                            "detailed_queries": {
                                "type": "array",
                                "description": "Specific search queries",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "required": ["depth", "detailed_queries"]
                    }
                }
            }
        ]

        payload = {
            "agent_config": {
                "agent_name": "Research Assistant",
                "description": "Expert in research with search capabilities",
                "system_prompt": "You are a research assistant with search capabilities.",
                "model_name": "gpt-4",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.7,
                "auto_generate_prompt": False,
                "tools_dictionary": tools_dictionary
            },
            "task": "Research the latest developments in quantum computing."
        }

        response = requests.post(
            f"{BASE_URL}/v1/agent/completions",
            headers=headers,
            json=payload
        )
        return response.json()
    ```

=== "cURL"

    ```bash title="tools_example.sh"
    curl -X POST "https://api.swarms.world/v1/agent/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "agent_config": {
          "agent_name": "Research Assistant",
          "description": "Expert in research with search capabilities",
          "system_prompt": "You are a research assistant with search capabilities.",
          "model_name": "gpt-4",
          "role": "worker",
          "max_loops": 1,
          "max_tokens": 8192,
          "temperature": 0.7,
          "auto_generate_prompt": false,
          "tools_dictionary": [
            {
              "type": "function",
              "function": {
                "name": "search_topic",
                "description": "Conduct an in-depth search on a topic",
                "parameters": {
                  "type": "object",
                  "properties": {
                    "depth": {
                      "type": "integer",
                      "description": "Search depth (1-3)"
                    },
                    "detailed_queries": {
                      "type": "array",
                      "description": "Specific search queries",
                      "items": {
                        "type": "string"
                      }
                    }
                  },
                  "required": ["depth", "detailed_queries"]
                }
              }
            }
          ]
        },
        "task": "Research the latest developments in quantum computing."
      }'
    ```

=== "TypeScript"

    ```typescript linenums="1" title="tools_example.ts"
    interface ToolFunction {
      name: string;
      description: string;
      parameters: {
        type: string;
        properties: {
          [key: string]: {
            type: string;
            description: string;
            items?: {
              type: string;
            };
          };
        };
        required: string[];
      };
    }

    interface Tool {
      type: string;
      function: ToolFunction;
    }

    interface AgentWithToolsConfig extends AgentConfig {
      tools_dictionary: Tool[];
    }

    interface AgentWithToolsPayload {
      agent_config: AgentWithToolsConfig;
      task: string;
    }

    async function runAgentWithTools() {
      const toolsDictionary: Tool[] = [
        {
          type: "function",
          function: {
            name: "search_topic",
            description: "Conduct an in-depth search on a topic",
            parameters: {
              type: "object",
              properties: {
                depth: {
                  type: "integer",
                  description: "Search depth (1-3)"
                },
                detailed_queries: {
                  type: "array",
                  description: "Specific search queries",
                  items: {
                    type: "string"
                  }
                }
              },
              required: ["depth", "detailed_queries"]
            }
          }
        }
      ];

      const payload: AgentWithToolsPayload = {
        agent_config: {
          agent_name: "Research Assistant",
          description: "Expert in research with search capabilities",
          system_prompt: "You are a research assistant with search capabilities.",
          model_name: "gpt-4",
          role: "worker",
          max_loops: 1,
          max_tokens: 8192,
          temperature: 0.7,
          auto_generate_prompt: false,
          tools_dictionary: toolsDictionary
        },
        task: "Research the latest developments in quantum computing."
      };

      try {
        const response = await axios.post(
          `${BASE_URL}/v1/agent/completions`,
          payload,
          {
            headers: {
              'x-api-key': API_KEY,
              'Content-Type': 'application/json'
            }
          }
        );
        return response.data;
      } catch (error) {
        console.error('Error:', error);
        throw error;
      }
    }
    ```

### Available Models

!!! info "Supported Models"

    Choose the right model for your use case:

    === "OpenAI"
        - `gpt-4`
        - `gpt-4o`
        - `gpt-4o-mini`

    === "Anthropic"
        - `claude-3-5-sonnet-20240620`
        - `claude-3-7-sonnet-latest`

    === "Groq"
        - `groq/llama3-70b-8192`
        - `groq/deepseek-r1-distill-llama-70b`

## Best Practices

!!! danger "Security"
    Never commit API keys or sensitive credentials to version control.

!!! warning "Rate Limits"
    Implement proper rate limiting and error handling in production.

!!! tip "Testing"
    Start with simple tasks and gradually increase complexity.

=== "Python"

    ```python linenums="1" title="best_practices.py"
    # Error Handling
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

    # Rate Limiting
    import time
    from tenacity import retry, wait_exponential

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    def make_api_call():
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response

    # Input Validation
    def validate_payload(payload):
        required_fields = ["agent_config", "task"]
        if not all(field in payload for field in required_fields):
            raise ValueError("Missing required fields")
    ```

=== "TypeScript"

    ```typescript linenums="1" title="best_practices.ts"
    // Error Handling
    try {
      const response = await axios.post(url, payload, { headers });
    } catch (error) {
      if (axios.isAxiosError(error)) {
        console.error('API Error:', error.response?.data);
      }
      throw error;
    }

    // Rate Limiting
    import { rateLimit } from 'axios-rate-limit';

    const http = rateLimit(axios.create(), { 
      maxRequests: 2,
      perMilliseconds: 1000
    });

    // Input Validation
    function validatePayload(payload: unknown): asserts payload is AgentPayload {
      if (!payload || typeof payload !== 'object') {
        throw new Error('Invalid payload');
      }

      const { agent_config, task } = payload as Partial<AgentPayload>;
      
      if (!agent_config || !task) {
        throw new Error('Missing required fields');
      }
    }
    ```

## Connect With Us

Join our community of agent engineers and researchers for technical support, cutting-edge updates, and exclusive access to world-class agent engineering insights!

| Platform | Description | Link |
|----------|-------------|------|
| 📚 Documentation | Official documentation and guides | [docs.swarms.world](https://docs.swarms.world) |
| 📝 Blog | Latest updates and technical articles | [Medium](https://medium.com/@kyeg) |
| 💬 Discord | Live chat and community support | [Join Discord](https://discord.gg/jM3Z6M9uMq) |
| 🐦 Twitter | Latest news and announcements | [@kyegomez](https://twitter.com/kyegomez) |
| 👥 LinkedIn | Professional network and updates | [The Swarm Corporation](https://www.linkedin.com/company/the-swarm-corporation) |
| 📺 YouTube | Tutorials and demos | [Swarms Channel](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ) |
| 🎫 Events | Join our community events | [Sign up here](https://lu.ma/5p2jnc2v) |
| 🚀 Onboarding Session | Get onboarded with Kye Gomez, creator and lead maintainer of Swarms | [Book Session](https://cal.com/swarms/swarms-onboarding-session) |
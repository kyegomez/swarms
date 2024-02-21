# `MajorityVoting` Documentation

## Overview

The `swarms.structs` library provides a flexible architecture for creating and managing swarms of agents capable of performing tasks and making decisions based on majority voting. This documentation will guide you through the `MajorityVoting` class, explaining its purpose, architecture, and usage with examples.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [The `MajorityVoting` Class](#the-majorityvoting-class)
  - [Class Definition](#class-definition)
  - [Parameters](#parameters)
  - [Methods](#methods)
    - [`__init__`](#__init__)
    - [`run`](#run)
- [Usage Examples](#usage-examples)
  - [Basic Usage](#basic-usage)
  - [Concurrent Execution](#concurrent-execution)
  - [Asynchronous Execution](#asynchronous-execution)
- [Advanced Features](#advanced-features)
- [Troubleshooting and FAQ](#troubleshooting-and-faq)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

The `swarms.structs` library introduces a mode of distributed computation through "agents" that collaborate to determine the outcome of tasks using a majority voting system. It becomes crucial in scenarios where collective decision-making is preferred over individual agent accuracy.

## Installation

To install the `swarms.structs` library, run the following command:

```bash
pip install swarms-structs
```

## The `MajorityVoting` Class

The `MajorityVoting` class is a high-level abstraction used to coordinate a group of agents that perform tasks and return results. These results are then aggregated to form a majority vote, determining the final output.

### Class Definition

### Parameters

| Parameter       | Type       | Default  | Description                                                          |
|-----------------|------------|----------|----------------------------------------------------------------------|
| agents          | List[Agent]| Required | A list of agent instances to participate in the voting process.      |
| concurrent      | bool       | False    | Enables concurrent execution using threading if set to `True`.      |
| multithreaded   | bool       | False    | Enables execution using multiple threads if set to `True`.          |
| multiprocess    | bool       | False    | Enables execution using multiple processes if set to `True`.        |
| asynchronous    | bool       | False    | Enables asynchronous execution if set to `True`.                    |
| output_parser   | callable   | None     | A function to parse the output from the majority voting function.   |
| autosave        | bool       | False    | Enables automatic saving of the process state if set to `True`. (currently not used in source code) |
| verbose         | bool       | False    | Enables verbose logging if set to `True`.                           |

### Methods

#### `__init__`

The constructor for the `MajorityVoting` class. Initializes a new majority voting system with the given configuration.

*This method doesn't return any value.*

#### `run`

Executes the given task by all participating agents and aggregates the results through majority voting.

| Parameter | Type      | Description                      |
|-----------|-----------|----------------------------------|
| task      | str       | The task to be performed.        |
| *args     | list      | Additional positional arguments. |
| **kwargs  | dict      | Additional keyword arguments.    |

*Returns:* List[Any] - The result based on the majority vote.

## Usage Examples

### Basic Usage

```python
from swarms.structs.agent import Agent
from swarms.structs.majority_voting import MajorityVoting


def create_agent(name):
    return Agent(name)


agents = [create_agent(name) for name in ["GPT-3", "Codex", "Tabnine"]]
majority_voting = MajorityVoting(agents)
result = majority_voting.run("What is the capital of France?")
print(result)  # Output: Paris
```

### Concurrent Execution

```python
majority_voting = MajorityVoting(agents, concurrent=True)
result = majority_voting.run("What is the largest continent?")
print(result)  # Example Output: Asia
```

### Asynchronous Execution

```python
majority_voting = MajorityVoting(agents, asynchronous=True)
result = majority_voting.run("What is the square root of 16?")
print(result)  # Output: 4
```


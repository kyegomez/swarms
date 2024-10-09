# Swarms CLI Documentation

The Swarms Command Line Interface (CLI) allows you to easily manage and run your Swarms of agents from the command line. This page will guide you through the installation process and provide a breakdown of the available commands.

## Installation

You can install the `swarms` package using `pip` or `poetry`.

### Using pip

```bash
pip3 install -U swarms
```

### Using poetry

```bash
poetry add swarms
```

Once installed, you can run the Swarms CLI with the following command:

```bash
poetry run swarms help
```

## Swarms CLI - Help

When running `swarms help`, you'll see the following output:

```
  _________                                     
 /   _____/_  _  _______ _______  _____   ______
 \_____  \ \/ \/ /\__  \_  __ \/     \ /  ___/
 /        \     /  / __ \|  | \/  Y Y  \___ \ 
/_______  / \/\_/  (____  /__|  |__|_|  /____  >
        \/              \/            \/     \/ 



    Swarms CLI - Help

    Commands:
    onboarding    : Starts the onboarding process
    help          : Shows this help message
    get-api-key   : Retrieves your API key from the platform
    check-login   : Checks if you're logged in and starts the cache
    read-docs     : Redirects you to swarms cloud documentation!
    run-agents    : Run your Agents from your agents.yaml

    For more details, visit: https://docs.swarms.world
```

### CLI Commands

Below is a detailed explanation of the available commands:

- **onboarding**  
  Starts the onboarding process to help you set up your environment and configure your agents.
  
  Usage:
  ```bash
  swarms onboarding
  ```

- **help**  
  Displays the help message, including a list of available commands.

  Usage:
  ```bash
  swarms help
  ```

- **get-api-key**  
  Retrieves your API key from the platform, allowing your agents to communicate with the Swarms platform.

  Usage:
  ```bash
  swarms get-api-key
  ```

- **check-login**  
  Verifies if you are logged into the platform and starts the cache for storing your login session.

  Usage:
  ```bash
  swarms check-login
  ```

- **read-docs**  
  Redirects you to the official Swarms documentation on the web for further reading.

  Usage:
  ```bash
  swarms read-docs
  ```

- **run-agents**  
  Executes your agents from the `agents.yaml` configuration file, which defines the structure and behavior of your agents.

  Usage:
  ```bash
  swarms run-agents
  ```

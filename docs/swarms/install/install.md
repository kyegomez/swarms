
# Swarms Installation Guide

<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://github.com/kyegomez/swarms/raw/master/images/swarmslogobanner.png"
      >
    </a>
  </p>
</div>

You can install `swarms` with pip in a
[**Python>=3.10**](https://www.python.org/) environment.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10 or higher: [Download Python](https://www.python.org/)
- pip (specific version recommended): `pip >= 21.0`
- git (for cloning the repository): [Download Git](https://git-scm.com/)

## Installation Options

=== "pip (Recommended)"

    #### Headless Installation

    The headless installation of `swarms` is designed for environments where graphical user interfaces (GUI) are not needed, making it more lightweight and suitable for server-side applications.

    ```bash
    pip install swarms
    ```

=== "Development Installation"

    === "Using virtualenv"

        1. **Clone the repository and navigate to the root directory:**

            ```bash
            git clone https://github.com/kyegomez/swarms.git
            cd swarms
            ```

        2. **Setup Python environment and activate it:**

            ```bash
            python3 -m venv venv
            source venv/bin/activate
            pip install --upgrade pip
            ```

        3. **Install Swarms:**

            - Headless install:

                ```bash
                pip install -e .
                ```

            - Desktop install:

                ```bash
                pip install -e .[desktop]
                ```

    === "Using Anaconda"

        1. **Create and activate an Anaconda environment:**

            ```bash
            conda create -n swarms python=3.10
            conda activate swarms
            ```

        2. **Clone the repository and navigate to the root directory:**

            ```bash
            git clone https://github.com/kyegomez/swarms.git
            cd swarms
            ```

        3. **Install Swarms:**

            - Headless install:

                ```bash
                pip install -e .
                ```

            - Desktop install:

                ```bash
                pip install -e .[desktop]
                ```

    === "Using Poetry"

        1. **Clone the repository and navigate to the root directory:**

            ```bash
            git clone https://github.com/kyegomez/swarms.git
            cd swarms
            ```

        2. **Setup Python environment and activate it:**

            ```bash
            poetry env use python3.10
            poetry shell
            ```

        3. **Install Swarms:**

            - Headless install:

                ```bash
                poetry install
                ```

            - Desktop install:

                ```bash
                poetry install --extras "desktop"
                ```

## Javascript

=== "NPM install (Work in Progress)"

    Get started with the NPM implementation of Swarms:

    ```bash
    npm install swarms-js
    ```

## Documentation

[Learn more about Swarms →](swarms/)

## Examples

Check out Swarms examples for building agents, data retrieval, and more.

[Checkout Swarms examples →](examples/)

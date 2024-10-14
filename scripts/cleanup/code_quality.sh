#!/bin/bash

# Navigate to the directory containing the 'tests' folder
# cd /path/to/your/code/directory

# Run autopep8 with max aggressiveness (-aaa) and in-place modification (-i)
# on all Python files (*.py) under the 'tests' directory.
autopep8 --in-place --aggressive --aggressive --recursive --experimental --list-fixes swarms/

# Run black with default settings, since black does not have an aggressiveness level.
# Black will format all Python files it finds in the 'tests' directory.
black .

# Run ruff on the 'tests' directory.
# Add any additional flags if needed according to your version of ruff.
ruff . --fix
ruff clean

# YAPF
yapf --recursive --in-place --verbose --style=google --parallel tests

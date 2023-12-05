#!/bin/bash

# Navigate to the directory containing the 'swarms' folder
# cd /path/to/your/code/directory

# Run autopep8 with max aggressiveness (-aaa) and in-place modification (-i)
# on all Python files (*.py) under the 'swarms' directory.
autopep8 --in-place --aggressive --aggressive --recursive --experimental --list-fixes swarms/

# Run black with default settings, since black does not have an aggressiveness level.
# Black will format all Python files it finds in the 'swarms' directory.
black --experimental-string-processing swarms/

# Run ruff on the 'swarms' directory.
# Add any additional flags if needed according to your version of ruff.
ruff --unsafe_fix

# YAPF
yapf --recursive --in-place --verbose --style=google --parallel swarms

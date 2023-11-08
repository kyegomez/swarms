#!/bin/bash

# Navigate to the directory containing the 'swarms' folder
# cd /path/to/your/code/directory

# Run autopep8 with max aggressiveness (-aaa) and in-place modification (-i)
# on all Python files (*.py) under the 'swarms' directory.
autopep8 --in-place --aggressive --aggressive --recursive --experimental swarms/

# Run black with default settings, since black does not have an aggressiveness level.
# Black will format all Python files it finds in the 'swarms' directory.
black --experimental-string-processing swarms/

# Run ruff on the 'swarms' directory.
# Add any additional flags if needed according to your version of ruff.
ruff swarms/

# If you want to ensure the script stops if any command fails, add 'set -e' at the top.

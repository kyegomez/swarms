#!/bin/bash

# Python 3.14t Free-Threaded Setup Script
# This script automates the installation of Python 3.14 free-threaded build with uv

set -e  # Exit on error

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    exit 1
fi

uv self update

uv python install 3.14t

uv venv --python 3.14t

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    :
else
    :
fi

uv pip install swarms 

echo "To run Python 3.14t directly without activating:"
echo "  uvx python@3.14t your_script.py"
echo ""
echo "=========================================="
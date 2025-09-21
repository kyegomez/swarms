#!/bin/bash

# Test script to verify Python detection logic
# This script tests the Python detection improvements made to setup.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

print_status "Testing Python detection logic..."

# Test Python detection (same logic as in setup.sh)
PYTHON_CMD=""

# Try to find Python command (check both python and python3)
if command_exists python3; then
    PYTHON_CMD="python3"
    print_status "Found python3 command"
elif command_exists python; then
    # Check if it's Python 3.x
    if python -c 'import sys; exit(0 if sys.version_info[0] == 3 else 1)' 2>/dev/null; then
        PYTHON_CMD="python"
        print_status "Found python command (Python 3.x)"
    else
        print_warning "Found python command but it's not Python 3.x"
    fi
fi

if [ -n "$PYTHON_CMD" ]; then
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_status "Found Python $PYTHON_VERSION using command: $PYTHON_CMD"
    
    # Check if Python version meets requirements (>=3.10)
    if $PYTHON_CMD -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)'; then
        print_success "Python version is compatible (>=3.10)"
    else
        print_error "Python 3.10 or higher is required. Current version: $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python is not installed or not found."
    print_error "Make sure Python is in your PATH and accessible as 'python' or 'python3'"
    exit 1
fi

print_success "Python detection test passed!"
print_status "Detected Python command: $PYTHON_CMD"
print_status "Python version: $PYTHON_VERSION"

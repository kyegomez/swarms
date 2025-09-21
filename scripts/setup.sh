#!/bin/bash

# Swarms Setup Script
# This script sets up the complete development environment for the Swarms project

set -e  # Exit on any error

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

print_status "Starting Swarms development environment setup..."

# Check Python version
print_status "Checking Python version..."
PYTHON_CMD=""

# Try to find Python command (check both python and python3)
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    # Check if it's Python 3.x
    if python -c 'import sys; exit(0 if sys.version_info[0] == 3 else 1)' 2>/dev/null; then
        PYTHON_CMD="python"
    fi
fi

if [ -n "$PYTHON_CMD" ]; then
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_status "Found Python $PYTHON_VERSION using command: $PYTHON_CMD"
    
    # Check if Python version meets requirements (>=3.10)
    if $PYTHON_CMD -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)'; then
        print_success "Python version is compatible (>=3.10)"
    else
        print_error "Python 3.10 or higher is required. Please install a compatible Python version."
        exit 1
    fi
else
    print_error "Python is not installed or not found. Please install Python 3.10 or higher."
    print_error "Make sure Python is in your PATH and accessible as 'python' or 'python3'"
    exit 1
fi

# Install Poetry if not present
if ! command_exists poetry; then
    print_status "Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | $PYTHON_CMD -
    
    # Add Poetry to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Check if installation was successful
    if command_exists poetry; then
        print_success "Poetry installed successfully"
    else
        print_error "Failed to install Poetry. Please install manually from https://python-poetry.org/"
        exit 1
    fi
else
    print_success "Poetry is already installed"
    poetry --version
fi

# Configure Poetry to create virtual environments in project directory
print_status "Configuring Poetry..."
poetry config virtualenvs.in-project true

# Check if the prefer-active-python option exists (available in newer Poetry versions)
if poetry config --list | grep -q "virtualenvs.prefer-active-python"; then
    poetry config virtualenvs.prefer-active-python true
    print_status "Set virtualenvs.prefer-active-python to true"
else
    print_warning "virtualenvs.prefer-active-python option not available in this Poetry version, skipping..."
fi

# Install dependencies
print_status "Installing project dependencies..."
poetry install --with dev,lint,test

print_success "All dependencies installed successfully"

# Activate virtual environment and run additional setup
print_status "Setting up development tools..."

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating .env file template..."
    cat > .env << 'EOF'
# Swarms Environment Variables
# Copy this file and fill in your actual values

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Other API Keys (add as needed)
# ANTHROPIC_API_KEY=your_anthropic_key_here
# COHERE_API_KEY=your_cohere_key_here

# Logging Level
LOG_LEVEL=INFO

# Development Settings
DEVELOPMENT=true
EOF
    print_success ".env template created - please fill in your API keys"
else
    print_status ".env file already exists"
fi

# Check if Git is initialized
if [ ! -d ".git" ]; then
    print_warning "Git repository not initialized. Run 'git init' if you want version control."
else
    print_success "Git repository detected"
fi

# Display virtual environment information
print_status "Virtual environment information:"
poetry env info

# Display installed packages
print_status "Installed packages:"
poetry show --tree

print_success "Setup completed successfully!"
print_status "Next steps:"
echo "  1. Activate the virtual environment: poetry shell"
echo "  2. Fill in your API keys in the .env file"
echo "  3. Run tests to verify installation: poetry run pytest"
echo "  4. Start developing!"

print_status "Useful commands:"
echo "  - poetry shell                    # Activate virtual environment"
echo "  - poetry run python <script>      # Run Python scripts"
echo "  - poetry run pytest               # Run tests"
echo "  - poetry add <package>            # Add new dependencies"
echo "  - poetry update                   # Update dependencies"

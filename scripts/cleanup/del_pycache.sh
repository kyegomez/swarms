#!/bin/bash

# Find and delete all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -r {} +

# Find and delete all .pyc files
find . -type f -name "*.pyc" -delete

# Find and delete all dist directories
find . -type d -name "dist" -exec rm -r {} +

# Find and delete all .ruff directories
find . -type d -name ".ruff" -exec rm -r {} +

# Find and delete all .egg-info directories
find . -type d -name "*.egg-info" -exec rm -r {} +

# Find and delete all .pyo files
find . -type f -name "*.pyo" -delete

# Find and delete all .pyd files
find . -type f -name "*.pyd" -delete

# Find and delete all .so files
find . -type f -name "*.so" -delete
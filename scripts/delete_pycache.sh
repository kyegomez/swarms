#!/bin/bash

# Find all __pycache__ directories and delete them
find . -type d -name "__pycache__" -exec rm -rf {} +
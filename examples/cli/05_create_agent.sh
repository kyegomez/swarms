#!/bin/bash

# Swarms CLI - Create Agent Example
# Create and run a custom agent

swarms agent \
    --name "Research Agent" \
    --description "AI research specialist" \
    --system-prompt "You are an expert research agent." \
    --task "Analyze current trends in renewable energy" \
    --model-name "gpt-4o-mini"


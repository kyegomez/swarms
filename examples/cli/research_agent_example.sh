#!/bin/bash

# Example: Research Agent with Auto Max Loops
# This creates a research agent that can autonomously loop until the task is complete

python3.12 -m swarms.cli.main agent \
  --name "Research Agent" \
  --description "An autonomous research agent that conducts thorough research and analysis" \
  --system-prompt "You are an expert research agent. Your role is to conduct comprehensive research, analyze information from multiple sources, synthesize findings, and provide well-structured research reports. You should be thorough, cite sources when possible, and ensure your research is accurate and complete." \
  --model-name "gpt-4.1" \
  --max-loops "auto" \


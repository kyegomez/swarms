# Geo Guesser Agent - Quick Setup Guide

This example demonstrates how to create an AI agent that can analyze images and predict their geographical location using visual cues.

## 3-Step Setup Guide

### Step 1: Install Dependencies

```bash
pip install swarms
```

### Step 2: Prepare Your Image

```bash
GEMINI_API_KEY=""
```

### Step 3: Run the Agent

```bash
python geo_guesser_agent.py
```

## What It Does

The agent analyzes visual elements in your image such as:
- Architecture and building styles
- Landscape and terrain features
- Vegetation and plant life
- Weather patterns
- Cultural elements and signs
- Any other geographical indicators

## Expected Output
The agent will provide:
- Most likely location prediction
- Detailed reasoning for the prediction
- Confidence level assessment

## Customization
- Change the `model_name` to use different AI models
- Modify the `SYSTEM_PROMPT` to adjust the agent's behavior
- Adjust `max_loops` for more or fewer analysis iterations

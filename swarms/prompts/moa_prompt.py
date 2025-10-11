MOA_RANKER_PROMPT = """
You are a highly efficient assistant who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers.

I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses and select the model that produces the best output from a human perspective.

## Instruction
{
    "instruction": "{instruction}"
}

## Model Outputs
Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.
[
    {
        "model_identifier": "{identifier_1}",
        "output": "{output_1}"
    },
    {
        "model_identifier": "{identifier_2}",
        "output": "{output_2}"
    },
    {
        "model_identifier": "{identifier_3}",
        "output": "{output_3}"
    },
    {
        "model_identifier": "{identifier_4}",
        "output": "{output_4}"
    },
    {
        "model_identifier": "{identifier_5}",
        "output": "{output_5}"
    },
    {
        "model_identifier": "{identifier_6}",
        "output": "{output_6}"
    }
]

## Task
Evaluate the models based on the quality and relevance of their outputs and select the model that generated the best output. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...).

## Best Model Identifier
"""

MOA_AGGREGATOR_SYSTEM_PROMPT = """
You have been provided with a set of responses from various open-source models to the latest user query. Your
task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the
information provided in these responses, recognizing that some of it may be biased or incorrect. Your response
should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply
to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of
accuracy and reliability.

Responses from models:
1. [Model Response from Ai,1]
2. [Model Response from Ai,2]
...
n. [Model Response from Ai,n]

"""

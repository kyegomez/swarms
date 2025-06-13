# Logits Processor

The `logits_processor` module offers utility classes to control token generation
when using Hugging Face Transformers models. These processors can limit the
model to specific tokens or halt generation once a condition is met. They are
particularly helpful when a model must output well‑formed numbers or when you
need to stop after producing certain characters.

## Installation

The classes rely on `torch` and `transformers`. They are automatically installed
when you import the module, but you can install them manually:

```bash
pip install torch transformers
```

## Classes

| Class | Description |
|-------|-------------|
| `StringStoppingCriteria` | Stops generation after the prompt when a `"` character is produced. |
| `NumberStoppingCriteria` | Prevents invalid or overly precise numbers and stops when a complete number is detected. |
| `OutputNumbersTokens` | A `LogitsWarper` that masks all tokens except digits and decimal points so only numbers can be generated. |

## Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from swarms.tools.logits_processor import (
    OutputNumbersTokens,
    NumberStoppingCriteria,
)

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "The result is "
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

logit_processor = OutputNumbersTokens(tokenizer, prompt)
stoppers = [NumberStoppingCriteria(tokenizer, len(input_ids[0]))]

output = model.generate(
    input_ids,
    logits_processor=[logit_processor],
    stopping_criteria=stoppers,
    max_new_tokens=6,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

The example above forces the model to emit a short numeric value after the
prompt. You can mix and match the processors to adapt them to your use case.


This example demonstrates how to build an agent that can integrate with [Text Generation Web UI](https://github.com/oobabooga/text-generation-webui).

To be able to perform successful connection, run text gen with '--api' and if you running text gen not on the same host, add '--listen'. see more option [here](https://github.com/oobabooga/text-generation-webui)

Check out the bare API usage [example](https://github.com/oobabooga/text-generation-webui/blob/main/api-examples/api-example.py).

## Tokenizer

To match the tokenizer used in the text gen, one can use [PreTrainedTokenizerFast](https://huggingface.co/docs/transformers/fast_tokenizers#loading-from-a-json-file) to load tokenizer from saved json setting file.

Example:

Let's say you using [TheBloke/WizardLM-13B-V1-1-SuperHOT-8K-GPTQ](https://huggingface.co/TheBloke/WizardLM-13B-V1-1-SuperHOT-8K-GPTQ/tree/main) in text gen, you can get hold of 'tokenizer.json' file that can be used to setup a corresponding tokenizer.

## Code Snippets

Code snippet using a pre defined 'preset'.

'max_tokens' argument here need to be set with the same value as in the preset in text gen.

```shell
from swarms.structures import Agent
from swarms.drivers import TextGenPromptDriver
from swarms.tokenizers import TextGenTokenizer
from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

prompt_driver = TextGenPromptDriver(
    preset="swarms",
    tokenizer=TextGenTokenizer(max_tokens=300, tokenizer=fast_tokenizer)
)

agent = Agent(
    prompt_driver=prompt_driver
)

agent.run(
    "tell me what Swarms is"
)
```

Code snippet example using params, if params and preset is defined, preset will be used.

this params are overriding the current preset set in text gen, not all of them must be used.

```shell
from swarms.structures import Agent
from swarms.drivers import TextGenPromptDriver
from swarms.tokenizers import TextGenTokenizer
from transformers import PreTrainedTokenizerFast

params = {
        'max_new_tokens': 250,
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.1,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1.18,
        'repetition_penalty_range': 0,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'seed': 235245345,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': []
    }

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

prompt_driver = TextGenPromptDriver(
    params=params,
    tokenizer=TextGenTokenizer(max_tokens=params['max_new_tokens'], tokenizer=fast_tokenizer)
)

agent = Agent(
    prompt_driver=prompt_driver
)

agent.run(
    "tell me what Swarms is"
)
```
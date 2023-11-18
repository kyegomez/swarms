---
base_model: Open-Orca/Mistral-7B-OpenOrca
datasets:
- Open-Orca/OpenOrca
inference: false
language:
- en
library_name: transformers
license: apache-2.0
model_creator: OpenOrca
model_name: Mistral 7B OpenOrca
model_type: mistral
pipeline_tag: text-generation
prompt_template: '<|im_start|>system

  {system_message}<|im_end|>

  <|im_start|>user

  {prompt}<|im_end|>

  <|im_start|>assistant

  '
quantized_by: TheBloke
---

<!-- header start -->
<!-- 200823 -->
<div style="width: auto; margin-left: auto; margin-right: auto">
<img src="https://i.imgur.com/EBdldam.jpg" alt="TheBlokeAI" style="width: 100%; min-width: 400px; display: block; margin: auto;">
</div>
<div style="display: flex; justify-content: space-between; width: 100%;">
    <div style="display: flex; flex-direction: column; align-items: flex-start;">
        <p style="margin-top: 0.5em; margin-bottom: 0em;"><a href="https://discord.gg/theblokeai">Chat & support: TheBloke's Discord server</a></p>
    </div>
    <div style="display: flex; flex-direction: column; align-items: flex-end;">
        <p style="margin-top: 0.5em; margin-bottom: 0em;"><a href="https://www.patreon.com/TheBlokeAI">Want to contribute? TheBloke's Patreon page</a></p>
    </div>
</div>
<div style="text-align:center; margin-top: 0em; margin-bottom: 0em"><p style="margin-top: 0.25em; margin-bottom: 0em;">TheBloke's LLM work is generously supported by a grant from <a href="https://a16z.com">andreessen horowitz (a16z)</a></p></div>
<hr style="margin-top: 1.0em; margin-bottom: 1.0em;">
<!-- header end -->

# Mistral 7B OpenOrca - GGUF
- Model creator: [OpenOrca](https://huggingface.co/Open-Orca)
- Original model: [Mistral 7B OpenOrca](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca)

<!-- description start -->
## Description

This repo contains GGUF format model files for [OpenOrca's Mistral 7B OpenOrca](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca).

<!-- description end -->
<!-- README_GGUF.md-about-gguf start -->
### About GGUF

GGUF is a new format introduced by the llama.cpp team on August 21st 2023. It is a replacement for GGML, which is no longer supported by llama.cpp.

Here is an incomplate list of clients and libraries that are known to support GGUF:

* [llama.cpp](https://github.com/ggerganov/llama.cpp). The source project for GGUF. Offers a CLI and a server option.
* [text-generation-webui](https://github.com/oobabooga/text-generation-webui), the most widely used web UI, with many features and powerful extensions. Supports GPU acceleration.
* [KoboldCpp](https://github.com/LostRuins/koboldcpp), a fully featured web UI, with GPU accel across all platforms and GPU architectures. Especially good for story telling.
* [LM Studio](https://lmstudio.ai/), an easy-to-use and powerful local GUI for Windows and macOS (Silicon), with GPU acceleration.
* [LoLLMS Web UI](https://github.com/ParisNeo/lollms-webui), a great web UI with many interesting and unique features, including a full model library for easy model selection.
* [Faraday.dev](https://faraday.dev/), an attractive and easy to use character-based chat GUI for Windows and macOS (both Silicon and Intel), with GPU acceleration.
* [ctransformers](https://github.com/marella/ctransformers), a Python library with GPU accel, LangChain support, and OpenAI-compatible AI server.
* [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), a Python library with GPU accel, LangChain support, and OpenAI-compatible API server.
* [candle](https://github.com/huggingface/candle), a Rust ML framework with a focus on performance, including GPU support, and ease of use.

<!-- README_GGUF.md-about-gguf end -->
<!-- repositories-available start -->
## Repositories available

* [AWQ model(s) for GPU inference.](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-AWQ)
* [GPTQ models for GPU inference, with multiple quantisation parameter options.](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GPTQ)
* [2, 3, 4, 5, 6 and 8-bit GGUF models for CPU+GPU inference](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF)
* [OpenOrca's original unquantised fp16 model in pytorch format, for GPU inference and for further conversions](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca)
<!-- repositories-available end -->

<!-- prompt-template start -->
## Prompt template: ChatML

```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant

```

<!-- prompt-template end -->


<!-- compatibility_gguf start -->
## Compatibility

These quantised GGUFv2 files are compatible with llama.cpp from August 27th onwards, as of commit [d0cee0d](https://github.com/ggerganov/llama.cpp/commit/d0cee0d36d5be95a0d9088b674dbb27354107221)

They are also compatible with many third party UIs and libraries - please see the list at the top of this README.

## Explanation of quantisation methods
<details>
  <summary>Click to see details</summary>

The new methods available are:
* GGML_TYPE_Q2_K - "type-1" 2-bit quantization in super-blocks containing 16 blocks, each block having 16 weight. Block scales and mins are quantized with 4 bits. This ends up effectively using 2.5625 bits per weight (bpw)
* GGML_TYPE_Q3_K - "type-0" 3-bit quantization in super-blocks containing 16 blocks, each block having 16 weights. Scales are quantized with 6 bits. This end up using 3.4375 bpw.
* GGML_TYPE_Q4_K - "type-1" 4-bit quantization in super-blocks containing 8 blocks, each block having 32 weights. Scales and mins are quantized with 6 bits. This ends up using 4.5 bpw.
* GGML_TYPE_Q5_K - "type-1" 5-bit quantization. Same super-block structure as GGML_TYPE_Q4_K resulting in 5.5 bpw
* GGML_TYPE_Q6_K - "type-0" 6-bit quantization. Super-blocks with 16 blocks, each block having 16 weights. Scales are quantized with 8 bits. This ends up using 6.5625 bpw

Refer to the Provided Files table below to see what files use which methods, and how.
</details>
<!-- compatibility_gguf end -->

<!-- README_GGUF.md-provided-files start -->
## Provided files

| Name | Quant method | Bits | Size | Max RAM required | Use case |
| ---- | ---- | ---- | ---- | ---- | ----- |
| [mistral-7b-openorca.Q2_K.gguf](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/blob/main/mistral-7b-openorca.Q2_K.gguf) | Q2_K | 2 | 3.08 GB| 5.58 GB | smallest, significant quality loss - not recommended for most purposes |
| [mistral-7b-openorca.Q3_K_S.gguf](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/blob/main/mistral-7b-openorca.Q3_K_S.gguf) | Q3_K_S | 3 | 3.16 GB| 5.66 GB | very small, high quality loss |
| [mistral-7b-openorca.Q3_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/blob/main/mistral-7b-openorca.Q3_K_M.gguf) | Q3_K_M | 3 | 3.52 GB| 6.02 GB | very small, high quality loss |
| [mistral-7b-openorca.Q3_K_L.gguf](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/blob/main/mistral-7b-openorca.Q3_K_L.gguf) | Q3_K_L | 3 | 3.82 GB| 6.32 GB | small, substantial quality loss |
| [mistral-7b-openorca.Q4_0.gguf](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/blob/main/mistral-7b-openorca.Q4_0.gguf) | Q4_0 | 4 | 4.11 GB| 6.61 GB | legacy; small, very high quality loss - prefer using Q3_K_M |
| [mistral-7b-openorca.Q4_K_S.gguf](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/blob/main/mistral-7b-openorca.Q4_K_S.gguf) | Q4_K_S | 4 | 4.14 GB| 6.64 GB | small, greater quality loss |
| [mistral-7b-openorca.Q4_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/blob/main/mistral-7b-openorca.Q4_K_M.gguf) | Q4_K_M | 4 | 4.37 GB| 6.87 GB | medium, balanced quality - recommended |
| [mistral-7b-openorca.Q5_0.gguf](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/blob/main/mistral-7b-openorca.Q5_0.gguf) | Q5_0 | 5 | 5.00 GB| 7.50 GB | legacy; medium, balanced quality - prefer using Q4_K_M |
| [mistral-7b-openorca.Q5_K_S.gguf](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/blob/main/mistral-7b-openorca.Q5_K_S.gguf) | Q5_K_S | 5 | 5.00 GB| 7.50 GB | large, low quality loss - recommended |
| [mistral-7b-openorca.Q5_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/blob/main/mistral-7b-openorca.Q5_K_M.gguf) | Q5_K_M | 5 | 5.13 GB| 7.63 GB | large, very low quality loss - recommended |
| [mistral-7b-openorca.Q6_K.gguf](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/blob/main/mistral-7b-openorca.Q6_K.gguf) | Q6_K | 6 | 5.94 GB| 8.44 GB | very large, extremely low quality loss |
| [mistral-7b-openorca.Q8_0.gguf](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/blob/main/mistral-7b-openorca.Q8_0.gguf) | Q8_0 | 8 | 7.70 GB| 10.20 GB | very large, extremely low quality loss - not recommended |

**Note**: the above RAM figures assume no GPU offloading. If layers are offloaded to the GPU, this will reduce RAM usage and use VRAM instead.



<!-- README_GGUF.md-provided-files end -->

<!-- README_GGUF.md-how-to-download start -->
## How to download GGUF files

**Note for manual downloaders:** You almost never want to clone the entire repo! Multiple different quantisation formats are provided, and most users only want to pick and download a single file.

The following clients/libraries will automatically download models for you, providing a list of available models to choose from:
- LM Studio
- LoLLMS Web UI
- Faraday.dev

### In `text-generation-webui`

Under Download Model, you can enter the model repo: TheBloke/Mistral-7B-OpenOrca-GGUF and below it, a specific filename to download, such as: mistral-7b-openorca.Q4_K_M.gguf.

Then click Download.

### On the command line, including multiple files at once

I recommend using the `huggingface-hub` Python library:

```shell
pip3 install huggingface-hub
```

Then you can download any individual model file to the current directory, at high speed, with a command like this:

```shell
huggingface-cli download TheBloke/Mistral-7B-OpenOrca-GGUF mistral-7b-openorca.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
```

<details>
  <summary>More advanced huggingface-cli download usage</summary>

You can also download multiple files at once with a pattern:

```shell
huggingface-cli download TheBloke/Mistral-7B-OpenOrca-GGUF --local-dir . --local-dir-use-symlinks False --include='*Q4_K*gguf'
```

For more documentation on downloading with `huggingface-cli`, please see: [HF -> Hub Python Library -> Download files -> Download from the CLI](https://huggingface.co/docs/huggingface_hub/guides/download#download-from-the-cli).

To accelerate downloads on fast connections (1Gbit/s or higher), install `hf_transfer`:

```shell
pip3 install hf_transfer
```

And set environment variable `HF_HUB_ENABLE_HF_TRANSFER` to `1`:

```shell
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download TheBloke/Mistral-7B-OpenOrca-GGUF mistral-7b-openorca.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
```

Windows Command Line users: You can set the environment variable by running `set HF_HUB_ENABLE_HF_TRANSFER=1` before the download command.
</details>
<!-- README_GGUF.md-how-to-download end -->

<!-- README_GGUF.md-how-to-run start -->
## Example `llama.cpp` command

Make sure you are using `llama.cpp` from commit [d0cee0d](https://github.com/ggerganov/llama.cpp/commit/d0cee0d36d5be95a0d9088b674dbb27354107221) or later.

```shell
./main -ngl 32 -m mistral-7b-openorca.Q4_K_M.gguf --color -c 2048 --temp 0.7 --repeat_penalty 1.1 -n -1 -p "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"
```

Change `-ngl 32` to the number of layers to offload to GPU. Remove it if you don't have GPU acceleration.

Change `-c 2048` to the desired sequence length. For extended sequence models - eg 8K, 16K, 32K - the necessary RoPE scaling parameters are read from the GGUF file and set by llama.cpp automatically.

If you want to have a chat-style conversation, replace the `-p <PROMPT>` argument with `-i -ins`

For other parameters and how to use them, please refer to [the llama.cpp documentation](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md)

## How to run in `text-generation-webui`

Further instructions here: [text-generation-webui/docs/llama.cpp.md](https://github.com/oobabooga/text-generation-webui/blob/main/docs/llama.cpp.md).

## How to run from Python code

You can use GGUF models from Python using the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) or [ctransformers](https://github.com/marella/ctransformers) libraries.

### How to load this model in Python code, using ctransformers

#### First install the package

Run one of the following commands, according to your system:

```shell
# Base ctransformers with no GPU acceleration
pip install ctransformers
# Or with CUDA GPU acceleration
pip install ctransformers[cuda]
# Or with AMD ROCm GPU acceleration (Linux only)
CT_HIPBLAS=1 pip install ctransformers --no-binary ctransformers
# Or with Metal GPU acceleration for macOS systems only
CT_METAL=1 pip install ctransformers --no-binary ctransformers
```

#### Simple ctransformers example code

```python
from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-OpenOrca-GGUF", model_file="mistral-7b-openorca.Q4_K_M.gguf", model_type="mistral", gpu_layers=50)

print(llm("AI is going to"))
```

## How to use with LangChain

Here are guides on using llama-cpp-python and ctransformers with LangChain:

* [LangChain + llama-cpp-python](https://python.langchain.com/docs/integrations/llms/llamacpp)
* [LangChain + ctransformers](https://python.langchain.com/docs/integrations/providers/ctransformers)

<!-- README_GGUF.md-how-to-run end -->

<!-- footer start -->
<!-- 200823 -->
## Discord

For further support, and discussions on these models and AI in general, join us at:

[TheBloke AI's Discord server](https://discord.gg/theblokeai)

## Thanks, and how to contribute

Thanks to the [chirper.ai](https://chirper.ai) team!

Thanks to Clay from [gpus.llm-utils.org](llm-utils)!

I've had a lot of people ask if they can contribute. I enjoy providing models and helping people, and would love to be able to spend even more time doing it, as well as expanding into new projects like fine tuning/training.

If you're able and willing to contribute it will be most gratefully received and will help me to keep providing more models, and to start work on new AI projects.

Donaters will get priority support on any and all AI/LLM/model questions and requests, access to a private Discord room, plus other benefits.

* Patreon: https://patreon.com/TheBlokeAI
* Ko-Fi: https://ko-fi.com/TheBlokeAI

**Special thanks to**: Aemon Algiz.

**Patreon special mentions**: Pierre Kircher, Stanislav Ovsiannikov, Michael Levine, Eugene Pentland, Andrey, 준교 김, Randy H, Fred von Graf, Artur Olbinski, Caitlyn Gatomon, terasurfer, Jeff Scroggin, James Bentley, Vadim, Gabriel Puliatti, Harry Royden McLaughlin, Sean Connelly, Dan Guido, Edmond Seymore, Alicia Loh, subjectnull, AzureBlack, Manuel Alberto Morcote, Thomas Belote, Lone Striker, Chris Smitley, Vitor Caleffi, Johann-Peter Hartmann, Clay Pascal, biorpg, Brandon Frisco, sidney chen, transmissions 11, Pedro Madruga, jinyuan sun, Ajan Kanaga, Emad Mostaque, Trenton Dambrowitz, Jonathan Leane, Iucharbius, usrbinkat, vamX, George Stoitzev, Luke Pendergrass, theTransient, Olakabola, Swaroop Kallakuri, Cap'n Zoog, Brandon Phillips, Michael Dempsey, Nikolai Manek, danny, Matthew Berman, Gabriel Tamborski, alfie_i, Raymond Fosdick, Tom X Nguyen, Raven Klaugh, LangChain4j, Magnesian, Illia Dulskyi, David Ziegler, Mano Prime, Luis Javier Navarrete Lozano, Erik Bjäreholt, 阿明, Nathan Dryer, Alex, Rainer Wilmers, zynix, TL, Joseph William Delisle, John Villwock, Nathan LeClaire, Willem Michiel, Joguhyik, GodLy, OG, Alps Aficionado, Jeffrey Morgan, ReadyPlayerEmma, Tiffany J. Kim, Sebastain Graf, Spencer Kim, Michael Davis, webtim, Talal Aujan, knownsqashed, John Detwiler, Imad Khwaja, Deo Leter, Jerry Meng, Elijah Stavena, Rooh Singh, Pieter, SuperWojo, Alexandros Triantafyllidis, Stephen Murray, Ai Maven, ya boyyy, Enrico Ros, Ken Nordquist, Deep Realms, Nicholas, Spiking Neurons AB, Elle, Will Dee, Jack West, RoA, Luke @flexchar, Viktor Bowallius, Derek Yates, Subspace Studios, jjj, Toran Billups, Asp the Wyvern, Fen Risland, Ilya, NimbleBox.ai, Chadd, Nitin Borwankar, Emre, Mandus, Leonard Tan, Kalila, K, Trailburnt, S_X, Cory Kujawski


Thank you to all my generous patrons and donaters!

And thank you again to a16z for their generous grant.

<!-- footer end -->

<!-- original-model-card start -->
# Original model card: OpenOrca's Mistral 7B OpenOrca


<p><h1>🐋 Mistral-7B-OpenOrca 🐋</h1></p>


![OpenOrca Logo](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca/resolve/main/Images/MistralOrcaLogo.png "MistralOrca Logo")
[<img src="https://raw.githubusercontent.com/OpenAccess-AI-Collective/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/OpenAccess-AI-Collective/axolotl)


# OpenOrca - Mistral - 7B - 8k

We have used our own [OpenOrca dataset](https://huggingface.co/datasets/Open-Orca/OpenOrca) to fine-tune on top of [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1).
This dataset is our attempt to reproduce the dataset generated for Microsoft Research's [Orca Paper](https://arxiv.org/abs/2306.02707).
We use [OpenChat](https://huggingface.co/openchat) packing, trained with [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl).

This release is trained on a curated filtered subset of most of our GPT-4 augmented data.
It is the same subset of our data as was used in our [OpenOrcaxOpenChat-Preview2-13B model](https://huggingface.co/Open-Orca/OpenOrcaxOpenChat-Preview2-13B).

**HF Leaderboard evals place this model as #2 for all models smaller than 30B at release time, outperforming all but one 13B model.**

This release provides a first: a fully open model with class-breaking performance, capable of running fully accelerated on even moderate consumer GPUs.
Our thanks to the Mistral team for leading the way here.

We affectionately codename this model: "*MistralOrca*"

If you'd like to try the model now, we have it running on fast GPUs unquantized: https://huggingface.co/spaces/Open-Orca/Mistral-7B-OpenOrca

Want to visualize our full (pre-filtering) dataset? Check out our [Nomic Atlas Map](https://atlas.nomic.ai/map/c1b88b47-2d9b-47e0-9002-b80766792582/2560fd25-52fe-42f1-a58f-ff5eccc890d2).

[<img src="https://huggingface.co/Open-Orca/OpenOrca-Preview1-13B/resolve/main/OpenOrca%20Nomic%20Atlas.png" alt="Atlas Nomic Dataset Map" width="400" height="400" />](https://atlas.nomic.ai/map/c1b88b47-2d9b-47e0-9002-b80766792582/2560fd25-52fe-42f1-a58f-ff5eccc890d2)


We are in-process with training more models, so keep a look out on our org for releases coming soon with exciting partners.

We will also give sneak-peak announcements on our Discord, which you can find here:

https://AlignmentLab.ai

or check the OpenAccess AI Collective Discord for more information about Axolotl trainer here:

https://discord.gg/5y8STgB3P3


# Quantized Models

Quantized versions of this model are generously made available by [TheBloke](https://huggingface.co/TheBloke).

- AWQ: https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-AWQ
- GPTQ: https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GPTQ
- GGUF: https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF


# Prompt Template

We used [OpenAI's Chat Markup Language (ChatML)](https://github.com/openai/openai-python/blob/main/chatml.md) format, with `<|im_start|>` and `<|im_end|>` tokens added to support this.

## Example Prompt Exchange

```
<|im_start|>system
You are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!
<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I am doing well!<|im_end|>
<|im_start|>user
Please tell me about how mistral winds have attracted super-orcas.<|im_end|>
```


# Evaluation

## HuggingFace Leaderboard Performance

We have evaluated using the methodology and tools for the HuggingFace Leaderboard, and find that we have dramatically improved upon the base model.
We find **105%** of the base model's performance on HF Leaderboard evals, averaging **65.33**.

At release time, this beats all 7B models, and all but one 13B.

![HF Leaderboard](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca/resolve/main/Images/MistralOrca7BHFLeaderboard.png)


| Metric | Value |
|-----------------------|-------|
| MMLU (5-shot)         | 61.73 |
| ARC (25-shot)         | 63.57 |
| HellaSwag (10-shot)   | 83.79 |
| TruthfulQA (0-shot)   | 52.24 |
| Avg.                  | 65.33 |

We use [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) to run the benchmark tests above, using the same version as the HuggingFace LLM Leaderboard.


## AGIEval Performance

We compare our results to the base Mistral-7B model (using LM Evaluation Harness).

We find **129%** of the base model's performance on AGI Eval, averaging **0.397**.
As well, we significantly improve upon the official `mistralai/Mistral-7B-Instruct-v0.1` finetuning, achieving **119%** of their performance.

![OpenOrca-Platypus2-13B AGIEval Performance](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca/resolve/main/Images/MistralOrca7BAGIEval.png "AGIEval Performance")

## BigBench-Hard Performance

We find **119%** of the base model's performance on BigBench-Hard, averaging **0.416**.

![OpenOrca-Platypus2-13B BigBench-Hard Performance](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca/resolve/main/Images/MistralOrca7BBigBenchHard.png "BigBench-Hard Performance")


# Dataset

We used a curated, filtered selection of most of the GPT-4 augmented data from our OpenOrca dataset, which aims to reproduce the Orca Research Paper dataset.


# Training

We trained with 8x A6000 GPUs for 62 hours, completing 4 epochs of full fine tuning on our dataset in one training run.
Commodity cost was ~$400.


# Citation

```bibtex
@software{lian2023mistralorca1
  title = {MistralOrca: Mistral-7B Model Instruct-tuned on Filtered OpenOrcaV1 GPT-4 Dataset},
  author = {Wing Lian and Bleys Goodson and Guan Wang and Eugene Pentland and Austin Cook and Chanvichet Vong and "Teknium"},
  year = {2023},
  publisher = {HuggingFace},
  journal = {HuggingFace repository},
  howpublished = {\url{https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca},
}
@misc{mukherjee2023orca,
      title={Orca: Progressive Learning from Complex Explanation Traces of GPT-4},
      author={Subhabrata Mukherjee and Arindam Mitra and Ganesh Jawahar and Sahaj Agarwal and Hamid Palangi and Ahmed Awadallah},
      year={2023},
      eprint={2306.02707},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@misc{longpre2023flan,
      title={The Flan Collection: Designing Data and Methods for Effective Instruction Tuning},
      author={Shayne Longpre and Le Hou and Tu Vu and Albert Webson and Hyung Won Chung and Yi Tay and Denny Zhou and Quoc V. Le and Barret Zoph and Jason Wei and Adam Roberts},
      year={2023},
      eprint={2301.13688},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

<!-- original-model-card end -->

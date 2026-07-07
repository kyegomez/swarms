# Model Examples

Examples demonstrating how to run Swarms agents against different model providers. Each subfolder is a **provider** (or model family) and contains runnable, self-contained scripts.

Set the matching API key as an environment variable before running any example (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `AZURE_API_KEY`, `GROQ_API_KEY`).

## Providers

### Anthropic
- [claude_4.py](anthropic/claude_4.py) — Claude 4 agent
- [claude_4_example.py](anthropic/claude_4_example.py) — Claude 4 integration
- [swarms_claude_example.py](anthropic/swarms_claude_example.py) — Swarms + Claude
- [fable_agent.py](anthropic/fable_agent.py) — Claude Fable 5 agent
- [fable_example.py](anthropic/fable_example.py) — Claude Fable 5 example

### Azure OpenAI
- [azure_agent.py](azure/azure_agent.py) — Azure OpenAI integration
- [azure_agent_api_verison.py](azure/azure_agent_api_verison.py) — Azure API version handling
- [azure_model_support.py](azure/azure_model_support.py) — Azure model support

### Cerebras
- [cerebras_example.py](cerebras/cerebras_example.py) — Cerebras model integration

### DeepSeek
- [deepseek_r1.py](deepseek/deepseek_r1.py) — DeepSeek R1 (native API)
- [fast_r1_groq.py](deepseek/fast_r1_groq.py) — DeepSeek R1 served via Groq
- [groq_deepseek_agent.py](deepseek/groq_deepseek_agent.py) — DeepSeek on Groq agent

### GPT-OSS
- [gpt_os_agent.py](gpt_oss/gpt_os_agent.py) — GPT-OSS agent
- [groq_gpt_oss_models.py](gpt_oss/groq_gpt_oss_models.py) — GPT-OSS on Groq
- [multi_agent_gpt_oss_example.py](gpt_oss/multi_agent_gpt_oss_example.py) — Multi-agent GPT-OSS

### Llama 4
- [llama_4.py](llama4/llama_4.py) — Llama 4 agent
- [litellm_example.py](llama4/litellm_example.py) — Llama 4 via LiteLLM
- [simple_agent.py](llama4/simple_agent.py) — Simple Llama 4 agent

### Lumo
- [lumo_example.py](lumo/lumo_example.py) — Lumo model integration

### Mistral
- [mistral_example.py](mistral/mistral_example.py) — Mistral integration

### Ollama
- [simple_example_ollama.py](ollama/simple_example_ollama.py) — Local models via Ollama

### OpenAI
- [gpt_5_example.py](openai/gpt_5_example.py) — GPT-5 integration
- [5.4_openai.py](openai/5.4_openai.py) — GPT-5.4 via Swarms client
- [4o_mini_demo.py](openai/4o_mini_demo.py) — GPT-4o mini demo
- [concurrent_gpt5.py](openai/concurrent_gpt5.py) — Concurrent GPT-5 processing
- [example_o3.py](openai/example_o3.py) — o3 reasoning example
- [o3_agent.py](openai/o3_agent.py) — o3 reasoning agent
- [reasoning_duo_batched.py](openai/reasoning_duo_batched.py) — Batched reasoning duo
- [test_async_litellm.py](openai/test_async_litellm.py) — Async LiteLLM test

### OpenRouter
- [glm_or.py](openrouter/glm_or.py) — GLM via OpenRouter
- [hy3_or.py](openrouter/hy3_or.py) — Hunyuan 3 via OpenRouter

### Qwen
- [qwen_3_base.py](qwen/qwen_3_base.py) — Qwen 3 base model

### vLLM
- [vllm_example.py](vllm/vllm_example.py) — vLLM integration
- [vllm_wrapper.py](vllm/vllm_wrapper.py) — vLLM wrapper

### Custom LLM
- [base_llm.py](custom_llm/base_llm.py) — Plug a custom LLM class into an `Agent`

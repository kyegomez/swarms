from swarms import Agent
from loguru import logger

# A large, STABLE system prompt. This is the prefix that gets cached — in a real
# app this would be your policy docs, style guide, schema, or long context that
# is identical on every call. Repeated to clear the provider's token minimum.
KNOWLEDGE_BASE = (
    "You are a senior financial analyst assistant. Follow these standing "
    "instructions on every response. "
    + (
        "Always ground claims in fundamentals: revenue growth, margins, free "
        "cash flow, balance-sheet health, and competitive moat. Flag "
        "assumptions explicitly. Prefer ranges over false precision. Never give "
        "personalized investment advice; provide analysis only. "
    )
    * 60  # repeat to build a large, cacheable prefix (well over 2k tokens)
)

agent = Agent(
    agent_name="CachedAnalyst",
    system_prompt=KNOWLEDGE_BASE,
    model_name="claude-sonnet-4-6",
    max_loops=1,
    prompt_caching=True,  # <-- the only flag you need
    persistent_memory=False,
    top_p=None,  # Anthropic rejects sending both temperature and top_p
)

# First call WRITES the prefix to the cache (billed as cache-creation tokens).
logger.info("=== First call (cache write) ===")
first = agent.run(
    "Give me a 3-bullet framework for evaluating a SaaS company."
)
print(first)

# Second call re-uses the cached prefix (billed as cheap cache-read tokens).
logger.info("=== Second call (cache hit) ===")
second = agent.run("Now do the same for a semiconductor company.")
print(second)

# To confirm caching actually happened, inspect the raw usage object on a direct
# wrapper call — look for `cache_creation_input_tokens` (first) and
# `prompt_tokens_details.cached_tokens` > 0 (second).
logger.info("=== Verifying cache usage via the LiteLLM wrapper ===")
from swarms.utils.litellm_wrapper import LiteLLM

llm = LiteLLM(
    model_name="claude-sonnet-4-6",
    system_prompt=KNOWLEDGE_BASE,
    prompt_caching=True,
    max_tokens=128,
    top_p=None,  # Anthropic rejects sending both temperature and top_p
    return_all=True,  # return the full response so we can read `usage`
)

for label in ("write", "read"):
    resp = llm.run(
        "Summarize your standing instructions in one sentence."
    )
    usage = resp["usage"] if isinstance(resp, dict) else resp.usage

    # Loguru logging to show caching result
    cache_success = False
    if (
        label == "write"
        and usage.get("cache_creation_input_tokens", 0) > 0
    ):
        cache_success = True
        logger.success(
            f"[{label}] Cache creation: {usage['cache_creation_input_tokens']} tokens written to cache."
        )
    elif (
        label == "read"
        and usage.get("prompt_tokens_details", {}).get(
            "cached_tokens", 0
        )
        > 0
    ):
        cache_success = True
        logger.success(
            f"[{label}] Cache hit detected: {usage['prompt_tokens_details']['cached_tokens']} tokens read from cache."
        )
    else:
        logger.warning(
            f"[{label}] Cache not used as expected. Usage details: {usage}"
        )

    print(f"[{label}] usage = {usage}")

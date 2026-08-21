# Feedo Protocol Integration

This directory provides an integration for the **Feedo Protocol**, a decentralized, end-to-end encrypted vector database designed specifically for AI agents.

By using `FeedoMemoryTools`, you can give your Swarms agents permanent, autonomous memory.

## Getting Started

1. **Get a Usage Key:**
   To interact with the Feedo network, you need a usage key. You can generate a free testnet key from the [Feedo Dashboard](https://feedo.ink).
   
   Once generated, set it as an environment variable:
   ```bash
   export FEEDO_USAGE_KEY="0x..."
   ```

2. **Install the SDK:**
   ```bash
   pip install feedo-sdk>=0.1.22
   ```

3. **Run the Example:**
   ```bash
   python feedo_memory_example.py
   ```

## Documentation

- **Feedo Website & Dashboard**: [feedo.ink](https://feedo.ink)
- **Feedo SDK Reference**: [PyPI: feedo-sdk](https://pypi.org/project/feedo-sdk/)

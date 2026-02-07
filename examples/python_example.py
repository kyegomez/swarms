#!/usr/bin/env python3
"""
Simple Python example showing basic import and inspection of the `swarms` package.

Run:
    python3 examples/python_example.py
"""

import swarms


def main() -> None:
    print("Imported swarms package successfully.")
    print("Available top-level modules (sample):")
    try:
        agents = [n for n in dir(swarms.agents) if not n.startswith("_")]
    except Exception as e:
        agents = [f"<error reading agents: {e}>"]
    try:
        tools = [n for n in dir(swarms.tools) if not n.startswith("_")]
    except Exception as e:
        tools = [f"<error reading tools: {e}>"]

    print("  Agents:", agents[:20])
    print("  Tools:", tools[:20])


if __name__ == "__main__":
    main()

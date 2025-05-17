from swarms.structs.agent import Agent
import xml.etree.ElementTree as ET

"""
User-Facing Example: How to get XML output from a Swarms agent
-------------------------------------------------------------
This example demonstrates how to use the Swarms agent framework to generate output in XML format.
You can use this pattern in your own projects to get structured, machine-readable results from any agent.
"""

if __name__ == "__main__":
    # Step 1: Create your agent and specify output_type="xml"
    agent = Agent(
        agent_name="XML-Output-Agent",
        agent_description="Agent that demonstrates XML output support",
        max_loops=2,
        model_name="gpt-4o-mini",
        dynamic_temperature_enabled=True,
        interactive=False,
        output_type="xml",  # Request XML output
    )

    # Step 2: Ask your question or give a task
    task = "Summarize the latest trends in AI research."
    xml_out = agent.run(task)

    # Step 3: Print the XML output for inspection or downstream use
    print("\n===== XML Output from Agent =====\n")
    print(xml_out)

    # Step 4: (Optional) Parse the XML output for programmatic use
    try:
        root = ET.fromstring(xml_out)
        print("\nParsed XML Root Tag:", root.tag)
        print("Number of top-level children:", len(root))
        # Print the first child tag and text for demonstration
        if len(root):
            print("First child tag:", root[0].tag)
            print("First child text:", root[0].text)
    except ET.ParseError as e:
        print(f"Failed to parse XML: {e}")

# You can copy and adapt this example for your own agent workflows!

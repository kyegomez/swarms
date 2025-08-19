#!/usr/bin/env python3
"""
Simple Hospital Agent Map Simulation Demo

A simplified version that demonstrates medical AI agents collaborating
on a headache case using only the .run() method.
"""

from swarms import Agent
from agent_map_simulation import AgentMapSimulation


def create_medical_agent(name, description, system_prompt):
    """
    Create a medical agent with basic configuration.

    Args:
        name: The agent's name
        description: Brief description of the agent's medical role
        system_prompt: The system prompt defining the agent's medical expertise

    Returns:
        Configured medical Agent instance
    """
    return Agent(
        agent_name=name,
        agent_description=description,
        system_prompt=system_prompt,
        model_name="gpt-4o-mini",
        output_type="str",
        max_loops=1,
    )


def main():
    """Run the simplified hospital simulation."""
    # Create simulation
    simulation = AgentMapSimulation(
        map_width=60.0,
        map_height=60.0,
        proximity_threshold=12.0,
        update_interval=3.0,
    )

    # Create medical agents
    agents = [
        create_medical_agent(
            "Dr.Sarah_Emergency",
            "Emergency Medicine physician",
            "You are Dr. Sarah, an Emergency Medicine physician specializing in rapid patient assessment and triage.",
        ),
        create_medical_agent(
            "Dr.Michael_Neuro",
            "Neurologist",
            "You are Dr. Michael, a neurologist specializing in headache disorders and neurological conditions.",
        ),
        create_medical_agent(
            "Nurse_Jennifer_RN",
            "Registered nurse",
            "You are Jennifer, an experienced RN specializing in patient care coordination and monitoring.",
        ),
        create_medical_agent(
            "Dr.Lisa_Radiology",
            "Diagnostic radiologist",
            "You are Dr. Lisa, a diagnostic radiologist specializing in neuroimaging and head imaging.",
        ),
    ]

    # Add agents to simulation
    for agent in agents:
        simulation.add_agent(
            agent=agent,
            movement_speed=1.5,
            conversation_radius=12.0,
        )

    # Medical case
    headache_case = """
    MEDICAL CONSULTATION - HEADACHE CASE
    
    Patient: 34-year-old female presenting to Emergency Department
    Chief complaint: "Worst headache of my life" - sudden onset 6 hours ago
    
    History: Sudden, severe headache (10/10 pain), thunderclap pattern
    Associated symptoms: Nausea, vomiting, photophobia, phonophobia
    Vitals: BP 145/92, HR 88, Temp 98.6°F, O2 99%
    
    Clinical concerns: Rule out subarachnoid hemorrhage, assess for secondary causes
    Team collaboration needed for emergent evaluation and treatment planning.
    """

    # Run simulation
    simulation.run(
        task=headache_case,
        duration=240,
        with_visualization=True,
        update_interval=3.0,
    )


if __name__ == "__main__":
    main()

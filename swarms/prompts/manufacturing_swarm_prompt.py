# Define detailed prompts for each agent
tasks = {
    "health_safety": (
        "Analyze the factory's working environment for health safety. Focus on"
        " cleanliness, ventilation, spacing between workstations, and personal"
        " protective equipment availability."
    ),
    "productivity": (
        "Review the factory's workflow efficiency, machine utilization, and"
        " employee engagement. Identify operational delays or bottlenecks."
    ),
    "safety": (
        "Analyze the factory's safety measures, including fire exits, safety"
        " signage, and emergency response equipment."
    ),
    "security": (
        "Evaluate the factory's security systems, entry/exit controls, and"
        " potential vulnerabilities."
    ),
    "sustainability": (
        "Inspect the factory's sustainability practices, including waste"
        " management, energy usage, and eco-friendly processes."
    ),
    "efficiency": (
        "Assess the manufacturing process's efficiency, considering the layout,"
        " logistics, and automation level."
    ),
}


# Define prompts for each agent
health_safety_prompt = tasks["health_safety"]
productivity_prompt = tasks["productivity"]
safety_prompt = tasks["safety"]
security_prompt = tasks["security"]
sustainability_prompt = tasks["sustainability"]
efficiency_prompt = tasks["efficiency"]


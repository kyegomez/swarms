from swarms import GodMode

god_mode = GodMode(num_workers=3, openai_api_key="", ai_name="Optimus Prime")
task = "What were the winning Boston Marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
god_mode.print_responses(task)
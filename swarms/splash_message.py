def splash():
	import json

	with open('config.json', 'r') as file:
		config = json.load(file)

	if config["display_message"]:
		config_display_message = config["display_message"] 
	else: 
		config["display_message"] = ""
		config_display_message = config["display_message"] 
		
	if config["version"]:	
		config_version = config["version"]
	else:
		config["version"] = ""
		config_version = config["version"]

	env_file_empty = config_display_message == ""

	if env_file_empty or config_display_message == "keep":
		print("""

Warning: Pytorch will install with a default cpu config that may be different then the one for your machine. if this causes errors, run: 
	pip uninstall pytorch 
	pip install pytorch
This message will only display the first time swarms is imported by default. 
Hit any key to continue, type "keep" | "yes" | "y" to keep showing this message, type "dismiss" to not show this message again
		    """)

			    
		continue_or_not = input("keep showing message? ")

		if (env_file_empty and continue_or_not not in ["yes", "y", "keep"]) or (config_display_message == "keep" and continue_or_not == "dismiss"):
			config["display_message"] = "dismiss"
		elif continue_or_not == "keep" or continue_or_not == None:
			config["display_message"] = "keep"

	with open('config.json', 'w') as file:
		json.dump(config, file, indent=4)
	
	print(f"version: {config_version}")
	print("some packages may not be installed correctly. see __init__.py for more details. ")
	print("importing swarms... ")

from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import history_output_formatter

# Create a simple conversation
conversation = Conversation()
conversation.add("user", "Hello")
conversation.add("assistant", "Hi there!")

# Export the conversation to YAML
yaml_output = history_output_formatter(conversation, type="yaml")
print("YAML Format:\n", yaml_output)

# Export the conversation to XML
xml_output = history_output_formatter(conversation, type="xml")
print("\nXML Format:\n", xml_output)

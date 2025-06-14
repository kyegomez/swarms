# history_output_formatter Utility

The `history_output_formatter` function converts a `Conversation` object into various formats.
It supports lists, dictionaries, strings, JSON, YAML, and XML.

## Export to YAML

```python
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import history_output_formatter

conversation = Conversation()
conversation.add("user", "Hello")
conversation.add("assistant", "Hi there!")

yaml_history = history_output_formatter(conversation, type="yaml")
print(yaml_history)
```

## Export to XML

```python
xml_history = history_output_formatter(conversation, type="xml")
print(xml_history)
```

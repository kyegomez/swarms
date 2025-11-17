import yaml
from typing import Any
from swarms.utils.xml_utils import to_xml_string
from swarms.utils.output_types import HistoryOutputType


def history_output_formatter(
    conversation: callable, type: HistoryOutputType = "list"
) -> Any:
    """
    Formats the output of a conversation object into various formats.

    Args:
        conversation (callable): The conversation object that provides various output methods.
        type (HistoryOutputType, optional): The desired output format.
            Supported values:
                - "list": Returns the conversation as a list of message dicts.
                - "dict" or "dictionary": Returns the conversation as a dictionary.
                - "string" or "str": Returns the conversation as a string.
                - "final" or "last": Returns the content of the final message.
                - "json": Returns the conversation as a JSON string.
                - "all": Returns the conversation as a string (same as "string").
                - "yaml": Returns the conversation as a YAML string.
                - "dict-all-except-first": Returns all messages except the first as a dictionary.
                - "list-final": Returns the final message as a list.
                - "str-all-except-first": Returns all messages except the first as a string.
                - "dict-final": Returns the final message as a dictionary.
                - "xml": Returns the conversation as an XML string.
            Defaults to "list".

    Returns:
        Union[List[Dict[str, Any]], Dict[str, Any], str]: The formatted conversation output.

    Raises:
        ValueError: If an invalid type is provided.
    """
    if type == "list":
        return conversation.return_messages_as_list()
    elif type in ["dict", "dictionary"]:
        return conversation.to_dict()
    elif type in ["string", "str"]:
        return conversation.get_str()
    elif type in ["final", "last"]:
        return conversation.get_final_message_content()
    elif type == "json":
        return conversation.to_json()
    elif type == "all":
        return conversation.get_str()
    elif type == "yaml":
        return yaml.safe_dump(conversation.to_dict(), sort_keys=False)
    elif type == "dict-all-except-first":
        return conversation.return_all_except_first()
    elif type == "list-final":
        return conversation.return_list_final()
    elif type == "str-all-except-first":
        return conversation.return_all_except_first_string()
    elif type == "dict-final":
        return conversation.return_dict_final()
    elif type == "xml":
        data = conversation.to_dict()
        return to_xml_string(data, root_tag="conversation")
    else:
        raise ValueError(f"Invalid type: {type}")

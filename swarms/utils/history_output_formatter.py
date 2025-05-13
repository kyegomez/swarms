import yaml
from swarms.structs.conversation import Conversation

from typing import Literal, Union, List, Dict, Any

HistoryOutputType = Literal[
    "list",
    "dict",
    "dictionary",
    "string",
    "str",
    "final",
    "last",
    "json",
    "all",
    "yaml",
    "xml",  # Added XML as a valid output type
    # "dict-final",
    "dict-all-except-first",
    "str-all-except-first",
]

output_type: HistoryOutputType


def history_output_formatter(
    conversation: Conversation, type: HistoryOutputType = "list"
) -> Union[List[Dict[str, Any]], Dict[str, Any], str]:
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
    elif type == "xml":
        from swarms.utils.xml_utils import to_xml_string
        return to_xml_string(conversation.to_dict(), root_tag="conversation")
    # elif type == "dict-final":
    #     return conversation.to_dict()
    elif type == "dict-all-except-first":
        return conversation.return_all_except_first()
    elif type == "str-all-except-first":
        return conversation.return_all_except_first_string()
    else:
        raise ValueError(f"Invalid type: {type}")

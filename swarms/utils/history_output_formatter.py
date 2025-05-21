import yaml
from swarms.structs.conversation import Conversation
from typing import Literal, Union, List, Dict, Any
from swarms.utils.xml_utils import to_xml_string

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
    "xml",
    # "dict-final",
    "dict-all-except-first",
    "str-all-except-first",
]


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
    elif type == "dict-all-except-first":
        return conversation.return_all_except_first()
    elif type == "str-all-except-first":
        return conversation.return_all_except_first_string()
    elif type == "xml":
        data = conversation.to_dict()
        return to_xml_string(data, root_tag="conversation")
    else:
        raise ValueError(f"Invalid type: {type}")

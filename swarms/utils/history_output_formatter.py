import yaml
from swarms.structs.conversation import Conversation


def history_output_formatter(
    conversation: Conversation, type: str = "list"
):
    if type == "list":
        return conversation.return_messages_as_list()
    elif type == "dict" or type == "dictionary":
        return conversation.to_dict()
    elif type == "string" or type == "str":
        return conversation.get_str()
    elif type == "final" or type == "last":
        return conversation.get_final_message_content()
    elif type == "json":
        return conversation.to_json()
    elif type == "all":
        return conversation.get_str()
    elif type == "yaml":
        return yaml.safe_dump(conversation.to_dict(), sort_keys=False)
    else:
        raise ValueError(f"Invalid type: {type}")

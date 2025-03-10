from swarms.structs.conversation import Conversation


def history_output_formatter(
    conversation: Conversation, type: str = "list"
):
    if type == "list":
        return conversation.return_messages_as_list()
    elif type == "dict":
        return conversation.to_dict()
    elif type == "string" or type == "str":
        return conversation.get_str()
    else:
        raise ValueError(f"Invalid type: {type}")

import json
from swarms.structs.malt import MALT

malt = MALT(
    max_loops=1,
    preset_agents=True,
)

malt.run(
    task="Please solve the following challenging mathematical problem: Prove that for any positive integer n, the sum of the first n positive odd integers equals n². Include a rigorous proof with clear steps and explanations."
)

with open("conversation_output.json", "w") as json_file:
    json.dump(
        malt.conversation.return_messages_as_dictionary(),
        json_file,
        indent=4,
    )

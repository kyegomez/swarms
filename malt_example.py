from swarms.structs.malt import MALT

malt = MALT(
    max_loops=1,
    preset_agents=True,
)

malt.run(
    task="Prove that the sum of the first n natural numbers is n(n+1)/2."
)

print(malt.conversation.return_json())

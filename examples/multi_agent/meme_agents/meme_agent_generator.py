from swarms.structs.meme_agent_persona_generator import (
    MemeAgentGenerator,
)


if __name__ == "__main__":
    example = MemeAgentGenerator(
        name="Meme-Swarm",
        description="A swarm of specialized AI agents collaborating on generating and sharing memes around cool media from 2001s",
        max_loops=1,
    )

    print(
        example.run(
            "Generate funny meme agents around cool media from 2001s"
        )
    )

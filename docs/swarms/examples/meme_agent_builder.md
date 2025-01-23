# Meme Agent Builder

- `pip3 install -U swarms`
-  Add your OpenAI API key to the `.env` file with `OPENAI_API_KEY=your_api_key`
-  Run the script
-  Multiple agents will be created and saved to the `meme_agents` folder
-  A swarm architecture will be selected autonomously and executed

```python
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

```

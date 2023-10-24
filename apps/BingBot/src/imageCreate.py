import discord
import asyncio
from src import log

logger = log.setup_logger(__name__)
using_func = {}

async def get_using_create(user_id):
    return using_func[user_id]
async def set_using_create(user_id, status: bool):
     using_func[user_id] = status

async def create_image(interaction: discord.Interaction, prompt: str, image_generator):
    await interaction.response.defer(ephemeral=False, thinking=True)
    using_func[interaction.user.id] = True
    try:
        embeds = []
        prompts = f"> **{prompt}** - <@{str(interaction.user.id)}> (***BingImageCreator***)\n\n"
        # Fetches image links 
        images = await image_generator.get_images(prompt)
        # Add embed to list of embeds
        [embeds.append(discord.Embed(url="https://www.bing.com/").set_image(url=image_link)) for image_link in images]
        await interaction.followup.send(prompts, embeds=embeds, wait=True)
    except asyncio.TimeoutError:
        await interaction.followup.send("> **Error: Request timed out.**")
        logger.exception("Error while create image: Request timed out.")
    except Exception as e:
        await interaction.followup.send(f"> **Error: {e}**")
        logger.exception(f"Error while create image: {e}")
    finally:
        using_func[interaction.user.id] = False
import discord
import re
from EdgeGPT.EdgeGPT import Chatbot, ConversationStyle
from src import log
from functools import partial

USE_SUGGEST_RESPONSES = True
logger = log.setup_logger(__name__)
using_func = {}


# To add suggest responses
class MyView(discord.ui.View):
    def __init__(
        self,
        interaction: discord.Interaction,
        chatbot: Chatbot,
        conversation_style: str,
        suggest_responses: list,
    ):
        super().__init__(timeout=120)
        self.button_author = interaction.user.id
        # Add buttons
        for label in suggest_responses:
            button = discord.ui.Button(label=label)

            # Button event
            async def callback(
                interaction: discord.Interaction,
                button_author: int,
                button: discord.ui.Button,
            ):
                if interaction.user.id != button_author:
                    await interaction.response.defer(ephemeral=True, thinking=True)
                    await interaction.followup.send(
                        "You don't have permission to press this button."
                    )
                elif not using_func[interaction.user.id]:
                    await interaction.response.defer(ephemeral=False, thinking=True)
                    # When click the button, all buttons will disable.
                    for child in self.children:
                        child.disabled = True
                    await interaction.followup.edit_message(
                        message_id=interaction.message.id, view=self
                    )
                    username = str(interaction.user)
                    usermessage = button.label
                    channel = str(interaction.channel)
                    logger.info(
                        f"\x1b[31m{username}\x1b[0m : '{usermessage}' ({channel}) [Style: {conversation_style}] [button]"
                    )
                    await send_message(
                        chatbot, interaction, usermessage, conversation_style
                    )
                else:
                    await interaction.response.defer(ephemeral=True, thinking=True)
                    await interaction.followup.send(
                        "Please wait for your last conversation to finish."
                    )

            self.add_item(button)
            self.children[-1].callback = partial(
                callback, button_author=self.button_author, button=button
            )


async def get_using_send(user_id):
    return using_func[user_id]


async def set_using_send(user_id, status: bool):
    using_func[user_id] = status


async def send_message(
    chatbot: Chatbot,
    interaction: discord.Interaction,
    user_message: str,
    conversation_style: str,
):
    using_func[interaction.user.id] = True
    reply = ""
    text = ""
    link_embed = ""
    images_embed = []
    all_url = []
    try:
        # Change conversation style
        if conversation_style == "creative":
            reply = await chatbot.ask(
                prompt=user_message,
                conversation_style=ConversationStyle.creative,
                simplify_response=True,
            )
        elif conversation_style == "precise":
            reply = await chatbot.ask(
                prompt=user_message,
                conversation_style=ConversationStyle.precise,
                simplify_response=True,
            )
        else:
            reply = await chatbot.ask(
                prompt=user_message,
                conversation_style=ConversationStyle.balanced,
                simplify_response=True,
            )

        # Get reply text
        text = f"{reply['text']}"
        text = re.sub(r"\[\^(\d+)\^\]", lambda match: "", text)

        # Get the URL, if available
        try:
            if len(reply["sources"]) != 0:
                for i, url in enumerate(reply["sources"], start=1):
                    if len(url["providerDisplayName"]) == 0:
                        all_url.append(f"{i}. {url['seeMoreUrl']}")
                    else:
                        all_url.append(
                            f"{i}. [{url['providerDisplayName']}]({url['seeMoreUrl']})"
                        )
            link_text = "\n".join(all_url)
            link_embed = discord.Embed(description=link_text)
        except:
            pass

        # Set the final message
        user_message = user_message.replace("\n", "")
        ask = f"> **{user_message}** - <@{str(interaction.user.id)}> (***style: {conversation_style}***)\n\n"
        response = f"{ask}{text}"

        # Discord limit about 2000 characters for a message
        while len(response) > 2000:
            temp = response[:2000]
            response = response[2000:]
            await interaction.followup.send(temp)

        # Get the image, if available
        try:
            if len(link_embed) == 0:
                all_image = re.findall("https?://[\w\./]+", str(reply["sources_text"]))
                [
                    images_embed.append(
                        discord.Embed(url="https://www.bing.com/").set_image(
                            url=image_link
                        )
                    )
                    for image_link in all_image
                ]
        except:
            pass
        # Add all suggest responses in list
        if USE_SUGGEST_RESPONSES:
            suggest_responses = reply["suggestions"]
            if images_embed:
                await interaction.followup.send(
                    response,
                    view=MyView(
                        interaction, chatbot, conversation_style, suggest_responses
                    ),
                    embeds=images_embed,
                    wait=True,
                )
            elif link_embed:
                await interaction.followup.send(
                    response,
                    view=MyView(
                        interaction, chatbot, conversation_style, suggest_responses
                    ),
                    embed=link_embed,
                    wait=True,
                )
            else:
                await interaction.followup.send(
                    response,
                    view=MyView(
                        interaction, chatbot, conversation_style, suggest_responses
                    ),
                    wait=True,
                )
        else:
            if images_embed:
                await interaction.followup.send(
                    response, embeds=images_embed, wait=True
                )
            elif link_embed:
                await interaction.followup.send(response, embed=link_embed, wait=True)
            else:
                await interaction.followup.send(response, wait=True)
    except Exception as e:
        await interaction.followup.send(f">>> **Error: {e}**")
        logger.exception(f"Error while sending message: {e}")
    finally:
        using_func[interaction.user.id] = False

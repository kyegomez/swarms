import discord
import re
import os
import json
import asyncio
from EdgeGPT.EdgeGPT import Chatbot, ConversationStyle
from dotenv import load_dotenv
from discord.ext import commands
from core.classes import Cog_Extension
from functools import partial
from src import log

load_dotenv()

USE_SUGGEST_RESPONSES = True
try:
    MENTION_CHANNEL_ID = int(os.getenv("MENTION_CHANNEL_ID"))
except:
    MENTION_CHANNEL_ID = None
logger = log.setup_logger(__name__)
sem = asyncio.Semaphore(1)
conversation_style = "balanced"

with open("./cookies.json", encoding="utf-8") as file:
    cookies = json.load(file)
chatbot = Chatbot(cookies=cookies)


# To add suggest responses
class MyView(discord.ui.View):
    def __init__(self, chatbot: Chatbot, suggest_responses: list):
        super().__init__(timeout=120)
        # Add buttons
        for label in suggest_responses:
            button = discord.ui.Button(label=label)

            # Button event
            async def callback(
                interaction: discord.Interaction, button: discord.ui.Button
            ):
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
                task = asyncio.create_task(
                    send_message(chatbot, interaction, usermessage)
                )
                await asyncio.gather(task)

            self.add_item(button)
            self.children[-1].callback = partial(callback, button=button)


# Show Dropdown
class DropdownView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=180)

        options = [
            discord.SelectOption(
                label="Creative",
                description="Switch conversation style to Creative",
                emoji="🎨",
            ),
            discord.SelectOption(
                label="Balanced",
                description="Switch conversation style to Balanced",
                emoji="⚖️",
            ),
            discord.SelectOption(
                label="Precise",
                description="Switch conversation style to Precise",
                emoji="🔎",
            ),
            discord.SelectOption(
                label="Reset", description="Reset conversation", emoji="🔄"
            ),
        ]

        dropdown = discord.ui.Select(
            placeholder="Choose setting", min_values=1, max_values=1, options=options
        )

        dropdown.callback = self.dropdown_callback
        self.add_item(dropdown)

    # Dropdown event
    async def dropdown_callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=False, thinking=True)
        if interaction.data["values"][0] == "Creative":
            await set_conversation_style("creative")
            await interaction.followup.send(
                f"> **Info: successfull switch conversation style to *{interaction.data['values'][0]}*.**"
            )
            logger.warning(
                f"\x1b[31mConversation style has been successfully switch to {interaction.data['values'][0]}\x1b[0m"
            )
        elif interaction.data["values"][0] == "Balanced":
            await set_conversation_style("balanced")
            await interaction.followup.send(
                f"> **Info: successfull switch conversation style to *{interaction.data['values'][0]}*.**"
            )
            logger.warning(
                f"\x1b[31mConversation style has been successfully switch to {interaction.data['values'][0]}\x1b[0m"
            )
        elif interaction.data["values"][0] == "Precise":
            await set_conversation_style("precise")
            await interaction.followup.send(
                f"> **Info: successfull switch conversation style to *{interaction.data['values'][0]}*.**"
            )
            logger.warning(
                f"\x1b[31mConversation style has been successfully switch to {interaction.data['values'][0]}\x1b[0m"
            )
        else:
            await chatbot.reset()
            await interaction.followup.send(f"> **Info: Reset finish.**")
            logger.warning("\x1b[31mBing has been successfully reset\x1b[0m")
        # disable dropdown after select
        for dropdown in self.children:
            dropdown.disabled = True
        await interaction.followup.edit_message(
            message_id=interaction.message.id, view=self
        )


# Set conversation style
async def set_conversation_style(style: str):
    global conversation_style
    conversation_style = style


async def set_chatbot(cookies):
    global chatbot
    chatbot = Chatbot(cookies=cookies)


async def send_message(chatbot: Chatbot, message, user_message: str):
    async with sem:
        if isinstance(message, discord.message.Message):
            await message.channel.typing()
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
            if isinstance(message, discord.interactions.Interaction):
                user_message = user_message.replace("\n", "")
                ask = f"> **{user_message}**\t(***style: {conversation_style}***)\n\n"
                response = f"{ask}{text}"
            else:
                response = f"{text}\t(***style: {conversation_style}***)"

            # Discord limit about 2000 characters for a message
            while len(response) > 2000:
                temp = response[:2000]
                response = response[2000:]
                if isinstance(message, discord.interactions.Interaction):
                    await message.followup.send(temp)
                else:
                    await message.channel.send(temp)

            # Get the image, if available
            try:
                if len(link_embed) == 0:
                    all_image = re.findall(
                        "https?://[\w\./]+", str(reply["sources_text"])
                    )
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

            if USE_SUGGEST_RESPONSES:
                suggest_responses = reply["suggestions"]
                if images_embed:
                    if isinstance(message, discord.interactions.Interaction):
                        await message.followup.send(
                            response,
                            view=MyView(chatbot, suggest_responses),
                            embeds=images_embed,
                            wait=True,
                        )
                    else:
                        await message.channel.send(
                            response,
                            view=MyView(chatbot, suggest_responses),
                            embeds=images_embed,
                        )
                elif link_embed:
                    if isinstance(message, discord.interactions.Interaction):
                        await message.followup.send(
                            response,
                            view=MyView(chatbot, suggest_responses),
                            embed=link_embed,
                            wait=True,
                        )
                    else:
                        await message.channel.send(
                            response,
                            view=MyView(chatbot, suggest_responses),
                            embed=link_embed,
                        )
                else:
                    if isinstance(message, discord.interactions.Interaction):
                        await message.followup.send(
                            response, view=MyView(chatbot, suggest_responses), wait=True
                        )
                    else:
                        await message.channel.send(
                            response, view=MyView(chatbot, suggest_responses)
                        )
            else:
                if images_embed:
                    if isinstance(message, discord.interactions.Interaction):
                        await message.followup.send(
                            response, embeds=images_embed, wait=True
                        )
                    else:
                        await message.channel.send(response, embeds=images_embed)
                elif link_embed:
                    if isinstance(message, discord.interactions.Interaction):
                        await message.followup.send(
                            response, embed=link_embed, wait=True
                        )
                    else:
                        await message.channel.send(response, embed=link_embed)
                else:
                    if isinstance(message, discord.interactions.Interaction):
                        await message.followup.send(response, wait=True)
                    else:
                        await message.channel.send(response)
        except Exception as e:
            if isinstance(message, discord.interactions.Interaction):
                await message.followup.send(f">>> **Error: {e}**")
            else:
                await message.channel.send(f">>> **Error: {e}**")
            logger.exception(f"Error while sending message: {e}")


class Event(Cog_Extension):
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author == self.bot.user:
            return
        if self.bot.user in message.mentions:
            if not MENTION_CHANNEL_ID or message.channel.id == MENTION_CHANNEL_ID:
                content = re.sub(r"<@.*?>", "", message.content).strip()
                if len(content) > 0:
                    username = str(message.author)
                    channel = str(message.channel)
                    logger.info(
                        f"\x1b[31m{username}\x1b[0m : '{content}' ({channel}) [Style: {conversation_style}]"
                    )
                    task = asyncio.create_task(send_message(chatbot, message, content))
                    await asyncio.gather(task)
                else:
                    await message.channel.send(view=DropdownView())
            elif MENTION_CHANNEL_ID is not None:
                await message.channel.send(
                    f"> **Can only be mentioned at <#{self.bot.get_channel(MENTION_CHANNEL_ID).id}>**"
                )


async def setup(bot):
    await bot.add_cog(Event(bot))

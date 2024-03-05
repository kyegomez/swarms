import discord
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Discord Bot Setup
client = discord.Client()

# AI Model Setup
tokenizer = AutoTokenizer.from_pretrained(
    "facebook/blenderbot-400M-distill"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/blenderbot-400M-distill"
)


@client.event
async def on_ready():
    print(f"Logged in as {client.user.name}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("!generate"):
        input = message.content[len("!generate") :]
        inputs = tokenizer(input, return_tensors="pt")
        outputs = model.generate(**inputs)
        generated_text = tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        await message.channel.send(generated_text[0])


client.run("YOUR_BOT_TOKEN")

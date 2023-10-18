import os
import discord
from discord.ext import commands
import interpreter
import dotenv
import whisper
from voice import transcribe

dotenv.load_dotenv(".env")

bot_token = os.getenv("DISCORD_TOKEN")
bot_id = os.getenv("BOT_ID")

# interpreter.api_key = os.getenv("API_KEY")
# interpreter.api_base = os.getenv("API_BASE")
# interpreter.auto_run = True


def split_text(text, chunk_size=1500):
    #########################################################################
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


# discord initial
intents = discord.Intents.all()
intents.message_content = True
client = commands.Bot(command_prefix="$", intents=intents)

message_chunks = []
send_image = False


@client.event
async def on_message(message):
    await client.process_commands(message)
    bot_mention = f'<@{bot_id}>'
    if (bot_mention not in message.content) or (message.author == client.user or message.content[0] == '$'):
        return
    response = []
    for chunk in interpreter.chat(message.content, display=False, stream=False):
        # await message.channel.send(chunk)
        if 'message' in chunk:
            response.append(chunk['message'])
    last_response = response[-1]

    max_message_length = 2000  # Discord's max message length is 2000 characters
    # Splitting the message into chunks of 2000 characters
    response_chunks = [last_response[i:i + max_message_length]
                       for i in range(0, len(last_response), max_message_length)]
    # Sending each chunk as a separate message
    for chunk in response_chunks:
        await message.channel.send(chunk)


@client.command()
async def join(ctx):
    if ctx.author.voice:
        channel = ctx.message.author.voice.channel
        print('joining..')
        await channel.connect()
        print('joined.')
    else:
        print("not in a voice channel!")


@client.command()
async def leave(ctx):
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
    else:
        print("not in a voice channel!")


@client.command()
async def listen(ctx):
    if ctx.voice_client:
        print('trying to listen..')
        ctx.voice_client.start_recording(discord.sinks.WaveSink(), callback, ctx)
        print('listening..')
    else:
        print("not in a voice channel!")

def transcribe(audio):

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result.text

async def callback(sink: discord.sinks, ctx):
    print('in callback..')
    for user_id, audio in sink.audio_data.items():
        if user_id == ctx.author.id:
            print('saving audio..')
            audio: discord.sinks.core.AudioData = audio
            print(user_id)
            filename = "audio.wav"
            with open(filename, "wb") as f:
                f.write(audio.file.getvalue())
            print('audio saved.')
            transcription = transcribe(filename)
            print(transcription)
            response = []
            for chunk in interpreter.chat(transcription, display=False, stream=True):
                # await message.channel.send(chunk)
                if 'message' in chunk:
                    response.append(chunk['message'])
            await ctx.message.channel.send(' '.join(response))

@client.command()
async def stop(ctx):
    ctx.voice_client.stop_recording()


@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")

client.run(bot_token)

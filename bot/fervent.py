import discord
from discord import Intents
from run import run

perms = Intents(messages=True)
client = discord.Client(intents=perms) # Set the permissions that this bot will have, right now only need it to send messages

@client.event
async def on_ready():
    print('Logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$hello'):
        await message.channel.send('Hello!')

run(client)

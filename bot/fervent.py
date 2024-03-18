import discord
from discord import Intents
from discord.ext import commands
from run import run

bot = commands.Bot(command_prefix="!", intents = Intents.all()) # Set the permissions that this bot will have, right now only need it to send messages

@bot.event
async def on_ready():
    print('Logged in as {0.user}'.format(bot))

@bot.event
async def on_message(message):
    if message.author == bot.user: # Don't let the bot respond to itself
        return
    if "hello" in message.content: 
        await message.reply("yo") # Directly reply to the author instead
run(bot)

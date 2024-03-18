import discord
from discord import Intents
from discord.ext import commands
from run import run
from transformers import pipeline


bot = commands.Bot(command_prefix="!", intents = Intents.all()) # Set the permissions that this bot will have, right now only need it to send messages
sentiment_pipeline = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment-latest") # Utilize roBERTa which was fine tuned on Twitter data

@bot.event
async def on_ready():
    print('Logged in as {0.user}'.format(bot)) # Indicate that this bot is ready


# @bot.command()
# async def setimage(ctx, sentiment: str, image_url: str):


@bot.event
async def on_message(message):
    if message.author == bot.user: # Don't let the bot respond to itself
        print("BOT MESSAGE", message.author)
        return
    data = [message.content] # Put in a list as the only item
    prediction = sentiment_pipeline(data)
    if prediction[0]['label'] == 'positive': # Set conditions to check what the prediction is
        await message.reply("positive prediction") # Directly reply to the author instead
    elif prediction[0]['label'] == 'negative':
        await message.reply("negative prediction")
    else:
        await message.reply("neutral prediction")
run(bot)

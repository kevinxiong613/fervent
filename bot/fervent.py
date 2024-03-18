from discord import Intents
from discord.ext import commands
from run import run
from transformers import pipeline
import time # Keep track of the time


bot = commands.Bot(command_prefix="!", intents = Intents.all()) # Set the permissions that this bot will have, right now only need it to send messages
sentiment_pipeline = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment-latest") # Utilize roBERTa which was fine tuned on Twitter data
previous_time = 0 # Keep track of how much time it has been since this bot has last sent an image

@bot.event
async def on_ready():
    print('Logged in as {0.user}'.format(bot)) # Indicate that this bot is ready

@bot.command() # Command in the form of "!setimage sentiment" but they also must have an image attached
async def setimage(ctx, sentiment: str):
    if ctx.message.author != ctx.guild.owner:
        await ctx.send("Only the server owner can set image configurations.")
        return
    print(sentiment)
    if sentiment != 'neutral' and sentiment != 'positive' and sentiment != 'negative': # Check if they are trying to upload an image for the incorrect sentiment
        await ctx.send("The only available sentiments in which you can upload images is 'positive', 'negative', and 'neutral. Please set one image for each of these sentiments only.")
        return
    image_list = ctx.message.attachments
    if len(image_list) != 1: # Check if they are trying upload more or less than one image
        await ctx.send("You must attach EXACTLY ONE image to set this sentiment's image configuration. No more no less.")
        return
    # Now we can start the image uploading logic

    image = image_list[0]
    await ctx.send("End")

@bot.event
async def on_message(message):
    global previous_time # By declaring previous_time as a global variable inside the on_message function using the global keyword, can now access and modify it across the entire script. 
    if message.author == bot.user: # Don't let the bot respond to itself
        return
    if len(message.content) >= 8 and "!setimage" == message.content[:9]: # Immediately process_commands if the leading characters are !setimage
        await bot.process_commands(message)
        return
    
    print(time.time())
    if time.time() - previous_time < 10: # Not enough time has elapsed to send another image. Send an image every 50 seconds
        return
    previous_time = time.time() # Otherwise we can record the current time again and send the sentiment image
    data = [message.content] # Put message contents in a list as the only item
    prediction = sentiment_pipeline(data)
    if prediction[0]['label'] == 'positive': # Set conditions to check what the prediction is
        await message.reply("positive prediction") # Directly reply to the author instead
    elif prediction[0]['label'] == 'negative':
        await message.reply("negative prediction")
    else:
        await message.reply("neutral prediction")
    
run(bot)

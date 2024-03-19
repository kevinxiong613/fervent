import io
import discord
from discord import Intents
from discord.ext import commands
from run import run, getBucketName, getKeys
from transformers import pipeline
import time # Keep track of the time
import boto3
import os
import requests


bot = commands.Bot(command_prefix="!", intents = Intents.all()) # Set the permissions that this bot will have, right now only need it to send messages
sentiment_pipeline = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment-latest") # Utilize roBERTa which was fine tuned on Twitter data
previous_time = 0 # Keep track of how much time it has been since this bot has last sent an image
sentiment_names = dict()

# Set up S3 client
ACCESS_KEY, SECRET_KEY = getKeys()
s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)


@bot.event
async def on_ready():
    print('Logged in as {0.user}'.format(bot)) # Indicate that this bot is ready

def upload_image_to_s3(image_url, sentiment):
    try:
        # Download image from URL
        response = requests.get(image_url)
        if response.status_code == 200:
            # Save image to a temporary file to upload to s3
            with open('app/temp_image.jpg', 'wb') as f:
                f.write(response.content)

            # Upload file to S3 bucket
            object_key = f"{sentiment}/{os.path.basename(image_url)}"
            s3.upload_file('app/temp_image.jpg', getBucketName(), object_key)
            sentiment_names[sentiment] = object_key # Save the sentiment's URL for message use later!
            print(sentiment_names)
            # Generate presigned URL for the uploaded file

            file_url = s3.generate_presigned_url('get_object', Params={'Bucket': getBucketName(), 'Key': object_key}) # Get the presigned URL
            return file_url
        else:
            print("Failed to download image from URL.")
            return None
    except Exception as e:
        print(f"Error uploading image to S3: {e}") # Something went wrong during upload to s3
        return None

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
    print(image.url)

    file_url = upload_image_to_s3(image.url, sentiment)
    
    if file_url:
        await ctx.send("Image uploaded successfully!")
    else:
        await ctx.send("Image not uploaded successfully!")

@bot.event
async def on_message(message):
    global previous_time # By declaring previous_time as a global variable inside the on_message function using the global keyword, can now access and modify it across the entire script. 
    if message.author == bot.user: # Don't let the bot respond to itself
        return
    if len(message.content) >= 8 and "!setimage" == message.content[:9]: # Immediately process_commands if the leading characters are !setimage
        await bot.process_commands(message)
        return
    
    if time.time() - previous_time < 2: # Not enough time has elapsed to send another image. Send an image every 50 seconds
        return
    previous_time = time.time() # Otherwise we can record the current time again and send the sentiment image
    data = [message.content] # Put message contents in a list as the only item
    prediction = sentiment_pipeline(data)
    if prediction[0]['label'] == 'positive': # Set conditions to check what the prediction is
        await message.reply("positive prediction") # Directly reply to the author instead
    elif prediction[0]['label'] == 'negative':
        try:
            response = s3.get_object(Bucket=getBucketName(), Key=sentiment_names['negative'])
            print(response)
            image_data = response['Body'].read()
            await message.reply("negative prediction", file=discord.File(io.BytesIO(image_data), filename='negative_image.png'))
        except Exception as e: # Error occured with retrieving the image
            print(f"Error retrieving image from S3: {e}")
            await message.reply("Error retrieving image.")
    else:
        await message.reply("neutral prediction")
    
run(bot)
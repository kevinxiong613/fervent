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

# Set up S3 client
ACCESS_KEY, SECRET_KEY = getKeys()
s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)


@bot.event
async def on_ready():
    print('Logged in as {0.user}'.format(bot)) # Indicate that this bot is ready

def upload_image_to_s3(guild_id, image_url, sentiment):
    try:
        # Download image from URL
        response = requests.get(image_url)
        if response.status_code == 200: # 200 means ok
            # Save image to a temporary file to upload to s3, save it to the same directory that fervent.py is in
            with open('app/temp_image.jpg', 'wb') as f:
                f.write(response.content)

            # Upload file to S3 bucket
            object_key = f"{guild_id}/{sentiment}.jpg" # Generate an object_key to query from s3 later of the form discord_server_id/sentiment/sentiment.jpg
            s3.delete_object(Bucket=getBucketName(), Key=object_key) # Deletes the previous picture used if it's already there

            s3.upload_file('app/temp_image.jpg', getBucketName(), object_key) # upload_file takes in 1. file path 2. Bucket name 3. Object key to query
            # Generate presigned URL for the uploaded file

            # Delete the local image file after upload
            os.remove('app/temp_image.jpg')

            file_url = s3.generate_presigned_url('get_object', Params={'Bucket': getBucketName(), 'Key': object_key}) # Get the presigned URL
            return file_url
        else:
            print("Failed to download image from URL.") # Indicates failure from downloading the image with the discord image.url link
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
    image_list = ctx.message.attachments # Grab the image_list from any attachments

    if len(image_list) != 1: # Check if they are trying upload more or less than one image
        await ctx.send("You must attach EXACTLY ONE image to set this sentiment's image configuration. No more no less.")
        return
    # Now we can start the image uploading logic
    image = image_list[0]

    file_url = upload_image_to_s3(ctx.guild.id, image.url, sentiment) # Send in the .url parameter associated with the image, the ID associated with the discord server, and the sentiment for this user's command
    
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
        try:
            response = s3.get_object(Bucket=getBucketName(), Key=f"{message.guild.id}/{'positive'}.jpg") # Retrieve a response from s3 for the positive stored image
            print(response)
            image_data = response['Body'].read() # Get the image data from the response body
            await message.reply(file=discord.File(io.BytesIO(image_data), filename='positive_image.png')) # Process the image data to be sent back
        except Exception as e: # Error occured with retrieving the image
            print(f"Error retrieving image from S3: {e}")
            # Don't print anything in this case, typically means an image hasn't been uploaded for this sentiment yet
    elif prediction[0]['label'] == 'negative': # If negative...
        try:
            response = s3.get_object(Bucket=getBucketName(), Key=f"{message.guild.id}/{'negative'}.jpg")
            print(response)
            image_data = response['Body'].read() # Get the image data from the response body
            await message.reply(file=discord.File(io.BytesIO(image_data), filename='negative_image.png'))
        except Exception as e:
            print(f"Error retrieving image from S3: {e}")
    else:
        try:
            response = s3.get_object(Bucket=getBucketName(), Key=f"{message.guild.id}/{'neutral'}.jpg")
            print(response)
            image_data = response['Body'].read() # Get the image data from the response body
            await message.reply(file=discord.File(io.BytesIO(image_data), filename='neutral_image.png'))
        except Exception as e:
            print(f"Error retrieving image from S3: {e}")
    
run(bot)

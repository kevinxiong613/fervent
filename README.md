## Fervent

• Created a Discord bot that reacts to messages based on sentiment, raising engagement in 70,000+ member servers
• Engineered a sentiment analysis model from scratch in Python with 91% accuracy on 500,000 messages, achieving 400%
faster speeds than RoBERTa and reducing the Docker image size to 8% of the size needed with PyTorch or TensorFlow
• Deployed bot on AWS ECS after containerizing with Docker and fetched reaction images via AWS S3 buckets

Example of setting the image:

![alt text](https://raw.githubusercontent.com/kevinxiong613/fervent/main/example1.png)

Example of the bot using sentiment analysis to send the correct corresponding reaction image:

![alt text](https://raw.githubusercontent.com/kevinxiong613/fervent/main/example2.png)

```text
Languages: Python
Technologies: AWS ECS, AWS S3, Docker, Pandas

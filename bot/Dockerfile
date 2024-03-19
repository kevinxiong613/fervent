# Use an official Python runtime as the base image, use slim for minimal packages
FROM python:3.9

# Set the working directory in the container to where the code itself is
WORKDIR /app

# Copy the requirements file in this directory into the container, current working directory of the Docker container.
COPY requirements.txt .

# Install dependencies
# --no-cache-dir: This option tells pip not to use the cache when installing packages. Using this option ensures that pip fetches the latest versions of packages and does not use any cached package information, which can sometimes lead to outdated or inconsistent installations.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Specify the command to run the application
CMD ["python", "fervent.py"]

# Some CLI stuff to reference later

# docker build -t myapp .  --> Builds a docker image using -t so that I can name it "myapp" or whatever I want, the "." indicates the build context and tells Docker to look for the Dockerfile and any other files required for the build process in the current directory. All files in the current directory and its subdirectories are included in the build context and can be referenced in the Dockerfile using relative paths.

# docker images --> See all docker images

# docker run --rm myapp -->  The command docker run --rm myapp runs a Docker container based on the Docker image named myapp, and the --rm flag ensures that the container is automatically removed when it exits.



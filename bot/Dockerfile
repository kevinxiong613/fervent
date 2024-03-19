# Use an official Python runtime as the base image, use slim for minimal packages
FROM python:3.9-slim AS builder

# Set the working directory in the container to where the code itself is
WORKDIR /app

# Copy the requirements file in this directory into the container, current working directory of the Docker container.
COPY requirements.txt .

# Install dependencies
# --no-cache-dir: This option tells pip not to use the cache when installing packages. Using this option ensures that pip fetches the latest versions of packages and does not use any cached package information, which can sometimes lead to outdated or inconsistent installations.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# Right now copies all files and directories from the app directory on the host machine into the /app directory within the Docker container.
COPY app/ .

# Specify the command to run the application
CMD ["python", "fervent.py"]


# Stage 2: Final Stage
# Final Stage
FROM python:3.9-slim

WORKDIR /app

# Copy only necessary files from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY --from=builder /app .

# Specify the command to run the application
CMD ["python", "fervent.py"]

# ------------- Multistage build saving space ----------
# Separation of Build and Runtime Dependencies: In a traditional Dockerfile, all dependencies required for building the application, such as compilers, development libraries, and build tools, are included in the final Docker image. However, these build-time dependencies are not needed at runtime and only add unnecessary bulk to the image.

# Discarding Unnecessary Artifacts: By using multi-stage builds, you can create a separate build stage (often referred to as the "builder" or "build" stage) where you install build-time dependencies, compile the application, and generate any necessary artifacts. Once the build process is complete, you can discard this intermediate stage, including all build-time dependencies and artifacts, by using a separate, slimmer base image for the final stage.

# Copying Only Necessary Files: With multi-stage builds, you can selectively copy only the necessary artifacts from the builder stage into the final stage. This ensures that only the compiled application code and runtime dependencies are included in the final image, while build-time dependencies and unnecessary build artifacts are left behind.

# Smaller Base Image: By using a slimmer base image (e.g., python:3.9-slim) for the final stage, you further reduce the size of the final Docker image. Slim base images typically contain only essential runtime components, resulting in a smaller image size compared to full-sized base images.

# ------------- Some CLI stuff to reference later -------------

# docker build -t myapp .  --> Builds a docker image using -t so that I can name it "myapp" or whatever I want, the "." indicates the build context and tells Docker to look for the Dockerfile and any other files required for the build process in the current directory. All files in the current directory and its subdirectories are included in the build context and can be referenced in the Dockerfile using relative paths.

# docker images --> See all docker images

# docker run --rm myapp -->  The command docker run --rm myapp runs a Docker container based on the Docker image named myapp, and the --rm flag ensures that the container is automatically removed when it exits. T



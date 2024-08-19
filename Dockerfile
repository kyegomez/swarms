
# ==================================
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /usr/src/swarms


# Install Python dependencies
# COPY requirements.txt and pyproject.toml if you're using poetry for dependency management
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install the 'swarms' package, assuming it's available on PyPI
RUN pip install -U swarms

# Copy the rest of the application
COPY . .

# Expose port if your application has a web interface
# EXPOSE 5000

# # Define environment variable for the swarm to work
# ENV OPENAI_API_KEY=your_swarm_api_key_here

# If you're using `CMD` to execute a Python script, make sure it's executable
# RUN chmod +x example.py

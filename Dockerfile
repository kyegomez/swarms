
# ==================================
# Use an official Python runtime as a parent image
FROM python:3.9-slim
RUN apt-get update && apt-get -y install libgl1-mesa-dev libglib2.0-0; apt-get clean
RUN pip install opencv-contrib-python-headless

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /usr/src/swarm_cloud


# Install Python dependencies
# COPY requirements.txt and pyproject.toml if you're using poetry for dependency management
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install the 'swarms' package, assuming it's available on PyPI
RUN pip install swarms

# Copy the rest of the application
COPY . .

# Add entrypoint script if needed
# COPY ./entrypoint.sh .
# RUN chmod +x /usr/src/swarm_cloud/entrypoint.sh

# Expose port if your application has a web interface
# EXPOSE 5000

# # Define environment variable for the swarm to work
# ENV SWARM_API_KEY=your_swarm_api_key_here

# # Add Docker CMD or ENTRYPOINT script to run the application
# CMD python your_swarm_startup_script.py
# Or use the entrypoint script if you have one
# ENTRYPOINT ["/usr/src/swarm_cloud/entrypoint.sh"]

# If you're using `CMD` to execute a Python script, make sure it's executable
# RUN chmod +x your_swarm_startup_script.py

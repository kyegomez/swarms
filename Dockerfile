# Use an official NVIDIA CUDA runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install Python, libgl1-mesa-glx and other dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* 

# Upgrade pip
RUN pip3 install --upgrade pip

# Install nltk
RUN pip install nltk

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt supervisor

# Create the necessary directory and supervisord.conf
RUN mkdir -p /etc/supervisor/conf.d && \
    echo "[supervisord] \n\
    nodaemon=true \n\
    [program:web_demo] \n\
    command=python3 web_demo.py \n\
    [program:host_local_tools] \n\
    command=python3 host_local_tools.py \n\
    " > /etc/supervisor/conf.d/supervisord.conf
# Make port 80 available to the world outside this container
EXPOSE 80

# Run supervisord when the container launches
CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf", "--port", "7860"]

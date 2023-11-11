
# Use an official NVIDIA CUDA runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

RUN apt update && apt install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx

# Install Python and other dependencies
RUN apt-get update && apt-get install libgl1 \
    apt-get update && apt-get install -y opencv-python-headless \
    apt-get install ffmpeg libsm6 libxext6  -y \
    pip3 install python3-opencv \
    apt-get -y install mesa-glx\
    pip install opencv-python-headless \
    apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    find /usr -name libGL.so.1 \
    ln -s /usr/lib/x86_64-linux-gnu/mesa/libGL.so.1 /usr/lib/libGL.so.1

# Upgrade pip
RUN pip3 install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt supervisor

# Create the necessary directory and supervisord.conf
RUN mkdir -p /etc/supervisor/conf.d && \
    echo "[supervisord] \n\
    nodaemon=true \n\
    [program:host_local_tools] \n\
    command=python3 host_local_tools.py \n\
    [program:web_demo] \n\
    command=python3 web_demo.py \n\
    " > /etc/supervisor/conf.d/supervisord.conf

# Make port 80 available to the world outside this container
EXPOSE 80

# Run supervisord when the container launches
CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

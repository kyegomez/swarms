# Use an official CUDA runtime as a parent image
FROM python

# Set the working directory in the container to /app
WORKDIR /app


# Install any needed packages specified in requirements.txt
#RUN apt-get update && apt-get install -y \
#    python3-pip \
#    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY scripts /app/scripts
COPY swarms /app/swarms
COPY example.py /app/example.py

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
# ENV NAME World

# Run app.py when the container launches
CMD ["python3", "example.py"]
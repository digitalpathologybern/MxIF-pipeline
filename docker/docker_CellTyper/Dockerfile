# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# It threw an error on setuptools, so ensure it is on the newest state
RUN pip install --upgrade pip setuptools wheel

# Install any needed packages specified in requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt
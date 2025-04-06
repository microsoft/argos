# Use the official Python 3.10 image based on Debian bookworm
FROM python:3.10-bookworm

# Set the working directory in the container
WORKDIR /argos

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Specify the command to run your application
CMD ["python3", "utility/test.py"]

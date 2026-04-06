# Stage 1: Use an official Python runtime as a parent image
# Using python:3.10-slim as it is a good balance of size and compatibility.
FROM python:3.10-slim 

# Stage 2: Set the working directory in the container
WORKDIR /app

# Stage 3: Set environment variables
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1
# Tells Flask which file to run and in which mode
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
# Pass API keys as build arguments for more security
ARG GROQ_API_KEY
ARG HF_TOKEN
ENV GROQ_API_KEY=${GROQ_API_KEY}
ENV HF_TOKEN=${HF_TOKEN}

# Stage 4: Copy and install the requirements
# Copy the requirements file first to leverage Docker's layer caching.
# If requirements.txt doesn't change, Docker won't reinstall the packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 5: Copy the rest of your application code into the container
# This includes your 'src' and 'Datasets' directories.
COPY . .

# Stage 6: Expose the port the app runs on
# This makes the port available to the host machine.
EXPOSE 5001

# Stage 7: Define the command to run the application
# Use Gunicorn for a production-ready web server instead of Flask's built-in one.
# It will run the 'app' instance from your 'app.py' file.
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]

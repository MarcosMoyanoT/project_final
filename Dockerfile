FROM python:3.10-slim

# The WORKDIR is correct. It creates a folder named 'app'
# and all subsequent commands will run from inside this folder.
WORKDIR /app

# Copy and install dependencies first for better caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Install gunicorn for the production server
RUN pip install gunicorn

# Copy all project files into the /app directory
COPY . .

# Expose the port that the container will listen on.
EXPOSE 8080

# This is the command that starts the web server.
# It tells gunicorn to run the 'app' object found in the 'api' module.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "api:app"]

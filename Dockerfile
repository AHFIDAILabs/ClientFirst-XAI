# Dockerfile for FastAPI Application: This Dockerfile sets up a FastAPI application with Uvicorn as the ASGI server and includes all necessary dependencies for running the application.

# Use a slim Python image for smaller size
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project directory into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI runs on (default is 8000)
EXPOSE 8000

# Command to run the FastAPI application with Uvicorn
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
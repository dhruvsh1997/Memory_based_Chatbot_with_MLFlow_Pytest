# Use a slim Python base
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (optional but useful for wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Work at a stable path that contains the project folder
WORKDIR /workspace

# Install Python dependencies first for better layer caching
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Copy the whole project
COPY . /workspace

# Optional: make sure Python can import the package-style path
ENV PYTHONPATH=/workspace

# Expose Flask port
EXPOSE 5000

# Default command runs the Flask app
CMD ["python", "MLFlow_Pytest_Chatbot/app.py"]

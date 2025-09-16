# Use official lightweight Python image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required for numpy, pandas, opencv, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port (Cloud Run expects 8080)
EXPOSE 8080

# Start FastAPI with Uvicorn
CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "8080"]

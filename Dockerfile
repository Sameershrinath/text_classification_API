# Use official Python base image
FROM python:3.10-slim

# Set work directory inside the container
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy all files (backend, models, etc.)
COPY . .

# Expose port for Uvicorn (FastAPI default)
EXPOSE 8000

# Run the app using start.sh (so we can make it executable)
CMD ["./start.sh"]

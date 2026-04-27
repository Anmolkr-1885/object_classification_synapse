# Use stable Python (IMPORTANT)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (important for numpy, sklearn, etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 10000

# Start app using gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
#Base Image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirment txt first for efficient caching
COPY requirements.txt .

# Intall all the dependencies from requirement.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backEnd Code
COPY backEnd/app ./app

# Set the Backend Port
EXPOSE 8000

# Run this command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

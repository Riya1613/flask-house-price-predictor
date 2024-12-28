FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the port the app will run on
EXPOSE 5000

# Run the Flask application
CMD ["python", "app.py"]


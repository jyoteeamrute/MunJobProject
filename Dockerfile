# Use an official Python runtime as a parent image
FROM python:3.12.2-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app/src
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK stopwords data
RUN python -m nltk.downloader stopwords

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "main.py"]

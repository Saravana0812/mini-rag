# use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# set the working directory in the container
WORKDIR /rag-app

# Copy the requirements file into the container at /rag-app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy the current directory contents into the container at /rag-app
COPY . /rag-app

# expose the port
EXPOSE 8080

# command run the application
CMD ["python", "-m", "streamlit", "run", "main.py"]

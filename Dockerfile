# Set up python version from base image
FROM python:3.12-slim

#  Set the working directory in the container
WORKDIR /code

# Copy the requirement file into the working directory
COPY ./requirements.txt /code/requirements.txt

# copy the scripts into the docker
COPY ./app /code/app

# secretes (do not fill them up!)
ENV WANDB_ORG=''
ENV WANDB_PROJECT=''
ENV WANDB_MODEL_NAME=''
ENV WANDB_MODEL_VERSION=''
ENV WANDB_API_KEY=''

# use port 8080
EXPOSE 8080

# run these commands when starting docker
CMD {"fastapi", "run", "app/main.py", "--port", "8080", "--reload"}

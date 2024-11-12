#!/bin/bash

docker build docker/ --tag="mnemo-agentic-demo:latest"

if [ $? -eq 0 ]; then
  echo "Docker build successful."
else
  echo "Docker build failed. Please check the messages above. Exiting..."
  exit 4
fi

# remove old container if it exists
docker container inspect mnemo-agentic-demo &>/dev/null && docker rm -f mnemo-agentic-demo

# Run docker container
docker run -p 8000:8000 --name mnemo-agentic-demo mnemo-agentic-demo:latest

if [ $? -eq 0 ]; then
  echo "Success! mnemo-agentic simple agent is running."
  echo "Go to http://localhost:8001 to access the mnemo-agentic simple agent."
else
  echo "MNEMO-agentic container failed to start. Please check the messages above."
fi
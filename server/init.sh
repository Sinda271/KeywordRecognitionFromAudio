#!/bin/bash

sudo apt-get update

# Install docker
sudo apt install docker.io

# start docker
sudo systemctl start docker
sudo systemctl enable  docker

# Install docker compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.27.4/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# build and run docker containers
# shellcheck disable=SC2164
cd ~/server
sudo docker-compose up --build


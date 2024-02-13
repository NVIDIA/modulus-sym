#!/bin/bash

# General info about new container / image
CONTAINER_NAME=modulus/norneressim
TAG=latest

# Build a container
docker build \
-t $CONTAINER_NAME .


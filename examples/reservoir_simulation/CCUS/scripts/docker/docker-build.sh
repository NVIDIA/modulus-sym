#!/bin/bash

# General info about new container / image
CONTAINER_NAME=modulus/ccusressim
TAG=latest

# Build a container
docker build \
-t $CONTAINER_NAME .


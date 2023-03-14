#!/bin/bash

docker build -f Dockerfile -t roble:latest .
docker tag roble:latest $USER/roble:latest
docker push $USER/roble:latest


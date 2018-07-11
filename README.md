
docker build . -t facerecognition

when docker is running on vm, add port forwarding of port 50000

docker run -it -p50000:8000 --name facerecognition facerecognition bin/bash 

cd facerecognition

python3 serve.py
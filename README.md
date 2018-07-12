
docker build . -t facerecognition

# when docker is running on vm, add port forwarding of port 50000

docker run -p50000:8000 --name facerecognition facerecognition

# go to http://localhost:50000 in your browser

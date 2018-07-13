
docker build . -t facerecognition

// when docker is running on a vm, add port forwarding of port 50000 to the vm

docker run -p50000:8000 --name facerecognition facerecognition

// go to http://localhost:50000 in your browser

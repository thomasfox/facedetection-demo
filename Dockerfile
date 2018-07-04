FROM python:3.6
RUN apt-get update
RUN apt-get install -y build-essential cmake libboost-all-dev python3-dev
RUN pip install dlib numpy scikit-image requests-toolbelt
COPY upload.py /upload.py
COPY serve.py /serve.py

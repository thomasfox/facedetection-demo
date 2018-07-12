FROM python:3.6
RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  libboost-all-dev \
  python3-dev
RUN pip install \
  dlib \
  numpy \
  requests-toolbelt \
  scikit-image
RUN mkdir /facerecognition \
  && chdir /facerecognition \
  && curl -O http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2 \
  && bzip2 -d shape_predictor_5_face_landmarks.dat.bz2 \
  && curl -O http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 \
  && bzip2 -d dlib_face_recognition_resnet_model_v1.dat.bz2
COPY *.py /facerecognition/
COPY *.css /facerecognition/
CMD [ "python3", "./facerecognition/serve.py" ]

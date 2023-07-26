##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-07-24 12:12:13 am
# @copyright MIT License
#

FROM python:3.8.10-slim-buster

LABEL Meet Patel 

WORKDIR /data

COPY . /data

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip --no-cache-dir install -r requirements.txt

RUN pip uninstall opencv-python -y

RUN pip install opencv_python_headless==4.8.0.74

EXPOSE 5000

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]

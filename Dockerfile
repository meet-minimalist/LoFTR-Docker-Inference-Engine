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

EXPOSE 5000

RUN "/data/setup.sh"

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]

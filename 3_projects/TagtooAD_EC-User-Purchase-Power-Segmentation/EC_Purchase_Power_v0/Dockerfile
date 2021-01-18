# This dockerfile uses the ubuntu image
# Author: abaoyang

# os
FROM ubuntu:18.04
MAINTAINER abaoyang jiunyi.yang@tagtoo.com

RUN apt-get update && apt-get -y upgrade

RUN apt-get install -y python3 python3-pip

WORKDIR /src

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY src .

RUN ["chmod", "+x", "run.sh"]

COPY application_default_credentials.json .

# ENTRYPOINT ["./run.sh"]


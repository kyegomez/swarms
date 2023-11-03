FROM python:3.8-slim-buster

WORKDIR /home/zack/code/swarms/*

ADD . /home/zack/code/swarms/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000


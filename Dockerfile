FROM python:3.8-slim-buster

WORKDIR /home/zack/code/

ADD . /home/zack/code/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000


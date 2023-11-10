FROM python:3.8-slim-buster

WORKDIR /usr/src/app

ADD . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

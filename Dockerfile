FROM python:3.7-python 

RUN pip install keras flask gunicorn

RUN mkdir /app

WORKDIR /app

FROM tensorflow/tensorflow:latest

RUN pip install keras flask gunicorn

RUN mkdir /app
COPY anomaly_service.py /app 
COPY all_models.pkl /app

WORKDIR /app

CMD ["gunicorn", "-b 0.0.0.0:8000", "anomaly_service:app"]

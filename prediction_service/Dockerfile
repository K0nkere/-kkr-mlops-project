FROM python:3.9.7-slim

WORKDIR /app

COPY requirements.txt requirements.txt

# ENV MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
# ENV AWS_DEFAULT_REGION=ru-central1
# ENV AWS_ACCESS_KEY_ID=<key_id>
# ENV AWS_SECRET_ACCESS_KEY=<key>
# ENV PUBLIC_SERVER_IP=84.201.155.166

RUN pip3 install -r requirements.txt

RUN pip3 install evidently

# EXPOSE 9696

COPY ["predict.py", "model_service.py", "./"]

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
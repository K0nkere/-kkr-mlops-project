FROM python:3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", ".env", "./"]

ENV MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
ENV AWS_DEFAULT_REGION=ru-central1
ENV AWS_ACCESS_KEY_ID=YCAJEwNTUy1TB_cku57IXUNjt
ENV AWS_SECRET_ACCESS_KEY=YCPbwZrpsvkpG4LnrVQwRCVIahXsRHAXq_o_FbIA
ENV PUBLIC_SERVER_IP=51.250.101.100

RUN pipenv install --system --deploy

COPY ["predict.py", "./"]

EXPOSE 5000
EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

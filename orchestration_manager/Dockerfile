FROM python:3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /manager

COPY [".env", "requirements.txt", "Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

# RUN pip install "uvicorn[standard]" gunicorn

COPY ["backend_services.py", "prefect_model.py", "prefect_monitoring_report.py","project-manager.py", "./"]

# ENV PUBLIC_SERVER_IP=51.250.27.103
# ENV MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
# ENV AWS_DEFAULT_REGION=ru-central1
# ENV AWS_ACCESS_KEY_ID=YCAJEwNTUy1TB_cku57IXUNjt
# ENV AWS_SECRET_ACCESS_KEY=YCPbwZrpsvkpG4LnrVQwRCVIahXsRHAXq_o_FbIA

EXPOSE 9898

# VOLUME /manager/targets:../targets

# ENTRYPOINT ["bash"]
# ENTRYPOINT ["gunicorn", "--workers=1","--worker-class=uvicorn.workers.UvicornWorker", "project-manager:app", "--bind=0.0.0.0:9898", "--timeout=600"]
ENTRYPOINT ["gunicorn", "project-manager:app", "--bind=0.0.0.0:9898"]

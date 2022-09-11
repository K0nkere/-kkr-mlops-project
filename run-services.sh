#!/usr/bin/env bash

docker-compose up --build -d

cd orchestration_manager

pipenv run prefect config set PREFECT_ORION_UI_API_URL="http://${PUBLIC_SERVER_IP}:4200/api"
pipenv run prefect config set PREFECT_API_URL="http://${PUBLIC_SERVER_IP}:4200/api"

pipenv run prefect orion start --host 0.0.0.0
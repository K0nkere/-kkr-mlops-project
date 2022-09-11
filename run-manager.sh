#!/usr/bin/env bash

cd orchestration_manager

pipenv run prefect deployment apply batch_analyze-deployment.yaml

pipenv run prefect agent start -q project-manager

pipenv run prefect agent start -q project-maneger
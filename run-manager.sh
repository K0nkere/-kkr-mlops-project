#!/usr/bin/env bash

cd orchestration_manager

pipenv run prefect deployment build prefect_monitoring_report.py:batch_analyze -n monitoring-report -q project-manager --cron "1 1 1 * *" 
pipenv run prefect deployment build prefect_model.py:retrain_request -n retrain-model -q project-manager --cron "5 1 1 * *"
pipenv run prefect deployment build prefect_model.py:main -n initial-train -q project-manager --cron "10 1 1 1 *"

pipenv run prefect deployment apply batch_analyze-deployment.yaml
pipenv run prefect deployment apply retrain_request-deployment.yaml
pipenv run prefect deployment apply main-deployment.yaml

pipenv run prefect agent start -q project-manager
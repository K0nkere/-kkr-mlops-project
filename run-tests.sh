#!/usr/bin/env bash

cd "$(dirname "$0")"

cd integration_tests

export PREDICTION_SERVICE_TEST="prediction-service"

docker build -t ${PREDICTION_SERVICE_TEST}:test ../3-deployment/.

docker-compose up --build -d

pipenv run pytest test_prediction.py

docker-compose down

if [ ${ERROR_CODE}!=0 ]; then
    docker-compose logs
fi

docker-compose down

cd ../orchestration_manager

pipenv run pytest ./tests/test_functions.py --disable-warnings

exit ${ERROR_CODE}
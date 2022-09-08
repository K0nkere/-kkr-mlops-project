Hello! This is a MLOps project based on the Kaggle Used Car Auction Prices dataset. The goal is to master the full lifetime cycle of ML model development. Project is fully created on the Yandex Cloud and covers:

    creation of a model (python, sklearn, xgboost)
    its tracking while training and validating (Yandex Cloud, s3-bucket, MLflow)
    orchestrating of retraining model and switching outdated model (Prefect)
    deployment of the best version as a web-service (docker, Flask) !!! - Online monitoring (Evidently, Prometeus, Graphana) !!! - unit and integration tests (pytest)

This guide contains all the instructions for reproducing the project, and the following links lead to a description of the main stages of creation.

Stage-0 Creation of cloud environment and preparation of original dataset 
Stage-1 Baseline models and pipelines 
Stage-2 Searching the best parameters with MLFlow tracking service 
Stage-3 Initial orchestrating with Prefect 
Stage-4 Deployment final model with Flask as a web-service 
!!! Stage-5 Monitoring 
!!! Stage-6 Tests

Deployment
insert correct values into .env
check ports for docker-compose is empty
    docker ps
        docker kill <container_id>
check ports for tunneling in VSCode is open
5001 for MLFlow
4200 for Prefect Orion
3000 for grafana


Error on
{
    "year": 2013,
    "make": "Mercedes-Benz",
    "model": "SL-Class",
    "trim": "SL550",
    "body": "Convertible",
    "transmission": NaN,
    "vin": "wddjk7da0df012717",
    "state": "pa",
    "condition": 3.5,
    "odometer": 11251.0,
    "color": NaN,
    "interior": NaN,
    "seller": "mercedes-benz financial services",
    "mmr": 72000,
    "saledate": "2015-05-28 02:00:00"
}
Hello! This is a MLOps project based on the Kaggle Used Car Auction Prices dataset. The goal is to master the full lifetime cycle of ML model development. Project is fully created on the Yandex Cloud and covers:

- creation of a model (python, sklearn, xgboost)
- its tracking while training and validating (Yandex Cloud, s3-bucket, MLflow)
- orchestrating of retraining model and switching outdated model (Prefect)
- deployment of the best version as a web-service (docker, Flask) 
- Online and batch monitoring in production (Evidently, Prometeus, Graphana)
- Unit and Integration tests
- Using best practices with pytest, pylint, black, isort and pre-commit hooks

This guide contains all the instructions for reproducing the project, and the following links lead to a description of the main stages of creation.

### Problem description
The dataset contains historical information about characteristics of used cars put up for various auctions and its selling prices. I want to create a web service will allows us to predict the initial price of a car on the basis of its type and information about recent transactions. 
Of course each car has its own characteristics and unique condition, nevertheless, service will allows a potential seller to evaluate his auto in the current market or projected price can be used by an auction as a starting point for further bidding.

This guide contains all the instructions for reproducing the project, and the following links lead to a description of the main stages of creation.

### Stage-0 Creation of cloud environment and preparation of original dataset
The dataset used for this project contains data from December'14 till July'15. Lets imitate the working of a web-service and imagine that the current data is 2014-5-30.
So for the training purposes i want to use data from December'14 till March'15 and April'15 will be the test. Load data function getting the current data and periods parameter equals number of months for training+validation&test.
I divided  the initial kaggle's data into files by month and upload these files in Yandex Cloud Object Storage (aws s3-bucket analog).
At the next stages of monitoring and orchestraring model on the latest data one can simply invoke the training scrip with parametes of current date and the desired number of periods and it will automaticly retrain the model considering the previous month as a test.

### Stage-1 Baseline models and pipelines
I used Ridge, Random Forest, XGBoost Regressions as a baseline. There are two steps of data preparation: dropping NA of important columns (mske, model, trim - that specifying what the car actually was) and training Column transformer, which imputes missing values. 
I was taking advantage of Hyperopt library in order to train each of the models. So it is returns the set of parameters that gives me the best model prediction on the train+valid part. After that, the best models are combined with preprocessor into a pipeline so i can save just a solid model into s3 bucket.
Unfortunetly training of Random Forest and XGBoost regressions usually take about 2 hours so for review I turned off these models 
(but you can turn on all of them:
- go to the folder **orchestration_manager**
- locate the section flow **def main()** and uncomment rows in the **model** variable
- model will be train for all models and select the best one of them on the certain period
)

Stage-2 Searching the best parameters with MLFlow tracking service 
Stage-3 Initial orchestrating with Prefect 
Stage-4 Deployment final model with Flask as a web-service 
Stage-5 Monitoring 
Stage-6 Tests

Deployment
insert correct values into .env
check ports for docker-compose is empty
    docker ps
        docker kill <container_id>
check ports for tunneling in VSCode is open
5001 for MLFlow
4200 for Prefect Orion
3000 for grafana
check for venv variables is setted after running run-venv.sh

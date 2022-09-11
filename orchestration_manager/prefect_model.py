import json
import os
from datetime import datetime
import requests

import boto3
import mlflow
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

BUCKET = os.getenv("BUCKET", 'kkr-mlops-zoomcamp')
PUBLIC_SERVER_IP = os.getenv("PUBLIC_SERVER_IP","127.0.0.1")

MLFLOW_TRACKING_URI = f"http://{PUBLIC_SERVER_IP}:5001/"
signal_url = "http://127.0.0.1:9898/manager"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow_client = MlflowClient(tracking_uri = MLFLOW_TRACKING_URI)

def return_pre_parent():
    path_to_script = os.path.split(__file__)[0]
    pre_parent = os.path.split(path_to_script)[0]

    return pre_parent

def read_file(key, bucket=BUCKET):
    """
    Reads file from bucket or local storage
    """
    try:
        session = boto3.session.Session()
        s3 = session.client(
            service_name='s3',
            endpoint_url='https://storage.yandexcloud.net',
            region_name='ru-central1',
            # aws_access_key_id = "id",
            # aws_secret_access_key = "key")
        )
        obj = s3.get_object(Bucket=bucket, Key=key)

        data = pd.read_csv(obj['Body'], sep=",")

    except:
        print(f"... Failed to connect to {BUCKET} bucket, getting data from local storage ...")
        file_path = return_pre_parent()
        data = pd.read_csv(f"{file_path}/{key}", sep=",", na_values='NaN')

    return data


@task
def load_data(current_date = "2015-5-17", periods = 1):

    dt_current = datetime.strptime(current_date, "%Y-%m-%d")

    if periods == 1:
        date_file = dt_current + relativedelta(months = - 1)
        print(f"... Getting TEST data for {date_file.year}-{date_file.month} period ...")
        test_data = read_file(key = f"datasets/car-prices-{date_file.year}-{date_file.month}.csv")

        return test_data

    else:
        train_data = pd.DataFrame()
        for i in range(periods+1, 1, -1):
            date_file = dt_current + relativedelta(months = - i)
            try:
                data = read_file(key = f"datasets/car-prices-{date_file.year}-{date_file.month}.csv")
                print(f"... Getting TRAIN data for {date_file.year}-{date_file.month} period ...")
            except:
                print(f"... Cannot find file car-prices-{date_file.year}-{date_file.month}.csv ...",
                    "using blank")
                data = None

            train_data = pd.concat([train_data, data])

        return train_data


@task
def na_filter(data):
    work_data = data.copy()
    non_type = work_data[data['make'].isna() | data['model'].isna() | data['trim'].isna()].index
    work_data.drop(non_type, axis=0, inplace=True)

    y = work_data.pop('sellingprice')

    return work_data, y


class FeaturesModifier:
    def __init__(self, columns):
        self.columns = columns

    def fit(self, _ = None):
        return self

    def transform(self, work_data, _ = None):

        work_data = pd.DataFrame(work_data, columns = self.columns)
        work_data['make_model_trim'] = work_data['make'] + '_'  + work_data['model'] + '_' + work_data['trim']
        work_data['year'] = work_data['year'].astype('str')

        cat_cols = ['year', 'make_model_trim', 'body', 'transmission', 'color', 'interior']
        num_cols = ['condition', 'odometer', 'mmr']

        X = work_data[cat_cols + num_cols].copy()
        X_dict = X.to_dict(orient = 'records')

        return X_dict

    def fit_transform(self, work_data, _ = None):
        return self.transform(work_data)


@task
def prepare_features(work_data, preprocessor = None):

    num_2_impute = ['condition', 'odometer', 'mmr']
    cat_2_impute = ['body', 'transmission']
    constant_2_impute = ['color', 'interior']
    others = ['year', 'make', 'model', 'trim']

    if not preprocessor:
        features_filler = ColumnTransformer([
            ('num_imputer', SimpleImputer(missing_values=np.nan, strategy='mean'), num_2_impute),
            ('cat_imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent'), cat_2_impute),
            ('cat_constant', SimpleImputer(missing_values=np.nan, strategy='most_frequent'), constant_2_impute),
            ('others', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='-1'), others )
            ]
        )

        fm = FeaturesModifier(columns = num_2_impute + cat_2_impute + constant_2_impute + others)

        dv = DictVectorizer()

        preprocessor = Pipeline(steps = [
            ('filler', features_filler),
            ('modifier', fm),
            ('dict_vectorizer', dv)
        ])

        X = preprocessor.fit_transform(work_data)

    else:
        X = preprocessor.transform(work_data)

    return X, preprocessor


@task
def params_search(train, valid, y_train, y_valid, train_dataset_period, models):

    best_models = []

    for baseline in models:

        mlflow.set_experiment(f"{baseline.__name__}-models")
        search_space = models[baseline]

        def objective(params):

            with mlflow.start_run():
                mlflow.set_tag("baseline", f"{baseline.__name__}")
                mlflow.log_param("train_dataset", train_dataset_period)
                mlflow.log_param("parameters", params)

                print('... Serching for the best parameters ... ')

                training_model = baseline(**params)
                training_model.fit(train, y_train)

                print('... Predicting on the valid dataset ...')
                prediction_valid = training_model.predict(valid)
                rmse_valid = mean_squared_error(y_valid, prediction_valid, squared = False)
                mae_valid = mean_absolute_error(y_valid, prediction_valid)
                mape_valid = mean_absolute_percentage_error(y_valid, prediction_valid)

                print(f'... Errors on valid: RMSE {rmse_valid} MAE {mae_valid} MAPE {mape_valid} ...', )
                mlflow.log_metric('rmse_valid', rmse_valid)
                mlflow.log_metric('mae_valid', mae_valid)
                mlflow.log_metric('mape_valid', mape_valid)

            return {'loss': mape_valid, 'status': STATUS_OK}

        best_result = fmin(fn = objective,
                    space = search_space,
                    algo = tpe.suggest,
                    max_evals = 20, # int(2**(len(models[baseline].items())-2)), #3,
                    trials = Trials(),
                    )

        print("... Best model ...\n", baseline(**space_eval(search_space, best_result)))
        best_models.append(baseline(**space_eval(search_space, best_result)))

        mlflow.end_run()

    return best_models


@task
def train_best_models(best_models_experiment, train, y_train, X_valid, y_valid, X_test, y_test, preprocessor, models, train_dataset_period):
# pylint: disable = too-many-arguments

    best_pipelines = []
    test_dataset_period = X_test["saledate"].max()[:7]
    query = f'parameters.train_dataset = "{train_dataset_period}"'

    mlflow.autolog()
    for model in models:

        experiment = mlflow.set_experiment(f"{model.__name__}-models")

        best_run = mlflow_client.search_runs(
                experiment_ids = experiment.experiment_id,
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results = 2,
                filter_string=query,
                order_by = ['metrics.mape_valid ASC']
            )

        print(f"... Training {model.__name__} with best params ...")

        mlflow.set_experiment(best_models_experiment) #"Auction-car-prices-best-models")

        with mlflow.start_run():

            mlflow.log_param("test_dataset", test_dataset_period)

            best_params = json.loads(best_run[0].data.params['parameters'].replace("'", "\""))
            staged_model = model(**best_params).fit(train, y_train)

            pipeline = Pipeline(
                steps = [
                    ('preprocessor', preprocessor),
                    ('model', staged_model)
                ]
            )
            predict_valid = pipeline.predict(X_valid)
            rmse_valid = mean_squared_error(y_valid, predict_valid, squared = False)

            predict_test = pipeline.predict(X_test)
            rmse_test = mean_squared_error(y_test, predict_test, squared = False)
            mae_test = mean_absolute_error(y_test, predict_test)
            mape_test = mean_absolute_percentage_error(y_test, predict_test)

            mlflow.log_metric("rmse_valid", rmse_valid)
            mlflow.log_metric("rmse_test", rmse_test)
            mlflow.log_metric('mae_test', mae_test)
            mlflow.log_metric('mape_test', mape_test)
            mlflow.sklearn.log_model(pipeline, artifact_path='full-pipeline')

            best_pipelines.append((model.__name__, pipeline))

            print("... {:} MODEL was saved as a RUN of {:} ...".format(model.__name__, best_models_experiment))

            mlflow.end_run()

    return best_pipelines


@task
def model_to_registry(best_models_experiment, model_name, test_dataset_period):

    experiment = mlflow.set_experiment(best_models_experiment) #'Auction-car-prices-best-models')

    query = f'parameters.test_dataset = "{test_dataset_period}"'
    best_model_run = mlflow_client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        filter_string=query,
        max_results=1,
        order_by=["metrics.mape_test ASC"]

        )
    RUN_ID = best_model_run[0].info.run_id
    model_uri = "runs:/{:}/full-pipeline".format(RUN_ID)

    print(f"... Registering model {model_name} ...")
    registered_model = mlflow.register_model(
            model_uri=model_uri,
            name = model_name
        )
    print(f"... Model RUN_ID {registered_model.run_id} was registered as version {registered_model.version} at {registered_model.current_stage} stage ...")

    return registered_model


@task
def model_promotion(current_date, model_name, registered_model_version, to_stage):

    promoted_model = mlflow_client.transition_model_version_stage(
                                name = model_name,
                                version = registered_model_version,
                                stage = to_stage,
                                archive_existing_versions=True
                                )

    mlflow_client.update_model_version(
        name = model_name,
        version = registered_model_version,
        description=f'The model was promoted to {to_stage} {current_date}'
        )
    print(f"... Model {model_name} version {registered_model_version} was promoted to {to_stage} {current_date} ...")

    return promoted_model


def load_model(model_name, stage=None):

    versions = mlflow_client.get_latest_versions(
                name=model_name,
                stages=[stage]
                )

    try:
        model_uri = f"models:/{model_name}/{versions[0].current_stage}"
        model = mlflow.pyfunc.load_model(model_uri = model_uri)

        return model, versions[0].version
    except:
        print(f"... There are no models at {stage} stage ...")

        return None, None


@task
def switch_model_of_production(X_test, y_test, model_name):

    staging_model, staging_version = load_model(model_name, stage = "Staging")
    if staging_model:
        staging_test_prediction = staging_model.predict(X_test)
        # rmse_staging = mean_squared_error(staging_test_prediction, y_test, squared=False)
        mape_staging = mean_absolute_percentage_error(staging_test_prediction, y_test)
        print(f"... MAPE={mape_staging} for the model version {staging_version} ...")
    else:
        mape_staging = np.inf

    production_model, production_version = load_model(model_name, stage = "Production")
    if production_model:
        production_test_prediction = production_model.predict(X_test)
        # rmse_production = mean_squared_error(production_test_prediction, y_test, squared=False)
        mape_production = mean_absolute_percentage_error(production_test_prediction, y_test)
        print(f"... MAPE={mape_production} for the model version {production_version} ...")

    else:
        mape_production = np.inf

    if mape_staging <= mape_production:
        print(f"... Need to switch models. Version {staging_version} is better than {production_version} ...")

        return staging_version

    else:
        print(f"... No need to switch models. Version {production_version} is the best ...")
        return None


@flow(task_runner = SequentialTaskRunner())
def main(current_date = "2015-5-21", periods = 5):

    best_models_experiment = "Auction-car-prices-best-models"
    model_name = "Auction-car-prices-prediction"

    train_data = load_data(current_date = current_date, periods = periods)
    X, y = na_filter(train_data)

    test_data = load_data(current_date = current_date)
    X_test, y_test = na_filter(test_data)

    train_dataset_period = X["saledate"].max()[:7]
    test_dataset_period = X_test["saledate"].max()[:7]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

    print("... Training preprocessor ...")
    train, preprocessor = prepare_features(X_train, preprocessor = None )
    valid, _  = prepare_features(X_valid, preprocessor)

    print("... Initializing parameters for baseline models ...")
    models = {
        # LinearRegression: {
        #     "fit_intercept": hp.choice("fit_intercept", ('True', 'False'))
        #     },
        Ridge: {"alpha": hp.loguniform("alpha", -5, 5),
                "fit_intercept": hp.choice("fit_intercept", ('True', 'False'))
            },
        # RandomForestRegressor: {
        #         'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        #         'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        #         'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        #         'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        #         'random_state': 42
        #         },
        # XGBRegressor: {
        #         'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        #         'learning_rate': hp.loguniform('learning_rate', -3, 0),
        #         'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        #         'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        #         'max_child_weight': hp.loguniform('max_child_weight', -1, 3),
        #         'num_boost_rounds': 100,
        #         # 'early_stopping_rounds': 20,
        #         'objective': 'reg:squarederror',
        #         'seed': 42,
        #         }
        }

    best_models = params_search(
        train, valid,
        y_train, y_valid,
        train_dataset_period,
        models)


    best_pipelines = train_best_models(
        best_models_experiment,
        train, y_train,
        X_valid, y_valid,
        X_test, y_test,
        preprocessor,
        models,
        train_dataset_period
        )

    registered_model = model_to_registry(best_models_experiment, model_name, test_dataset_period)

    model_promotion(
        current_date,
        model_name,
        registered_model_version=registered_model.version,
        to_stage = "Staging"
        )

    switch_to_version = switch_model_of_production(X_test, y_test, model_name)

    if switch_to_version:
        model_promotion(
            current_date = current_date,
            model_name = model_name,
            registered_model_version = switch_to_version,
            to_stage="Production"
            )

    else:
        print("... Current model is OK ...")

@flow
def retrain_request():

    print("... Sending a check for model retraining ...")
    check = {
        "service": "training_model"
    }

    response = requests.post(
                    url=signal_url,
                    json=check,
                    timeout=60
                )

    response = response.json()

    current_date = response["current_date"]

    if response["current_date"]:
        current_date = response["current_date"]

        print(f"... Retraining model with data up to {current_date}")
        main(current_date=current_date)
    else:
        print("... No request to retrain model ...")


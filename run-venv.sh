#!/usr/bin/env bash

cd integration_tests
pipenv install -r requirements.txt --dev deepdiff --python=3.9

cd ../orchestration_manager
pipenv install -r requirements.txt --dev deepdiff --python=3.9

cd ..
setup:
	cp my.env ./orchestration_manager/.env
	bash run-venv.sh

test:
	bash run-tests.sh

quality_checks:
	isort .
	black .
	pylint --recursive=y --exit-zero .

preparation: tests quality_checks
	echo ${PUBLIC_SERVER_IP}
	echo ${MLFLOW_S3_ENDPOINT_URL}
	echo ${AWS_DEFAULT_REGION}
	echo ${BACKEND_URI}
	echo ${ARTIFACT_ROOT}
	bash run-services.sh


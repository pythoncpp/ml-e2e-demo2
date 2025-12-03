.PHONY: install test train deploy clean

install:
	pip3 install -r requirements/requirements.txt
	pip3 install -r requirements/requirements-dev.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

train:
	python3 scripts/train_pipeline.py

lint:
	flake8 src tests

run-api:
	python3 app/api.py

run-mlflow:
	mlflow server --host 0.0.0.0 --port 8000 

docker-build:
	docker image build -f deployment/Dockerfile -t amitksunbeam/salary-predictor:latest .

docker-run:
	docker run -p 9000:9000 amitksunbeam/salary-predictor:latest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm train_test.csv


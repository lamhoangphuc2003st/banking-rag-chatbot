.PHONY: install install-dev api web lint test eval docker-up docker-down migrate crawl normalize chunk index

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev,eval,scraping]"

api:
	uvicorn apps.api.app.main:app --host 0.0.0.0 --port 8000 --reload

web:
	cd apps/web && npm run dev

lint:
	ruff check .
	mypy apps packages

test:
	pytest

eval:
	python -m packages.evals.retrieval_eval --golden data/golden/retrieval_golden.jsonl

docker-up:
	docker compose up --build

docker-down:
	docker compose down

migrate:
	alembic -c apps/api/alembic.ini upgrade head

crawl:
	python -m packages.data_pipeline.cli crawl

normalize:
	python -m packages.data_pipeline.cli normalize

chunk:
	python -m packages.data_pipeline.cli chunk

index:
	python -m packages.data_pipeline.cli index

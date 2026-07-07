.PHONY: install install-dev api web lint test eval eval-guardrails verify-runtime verify-runtime-external docker-up docker-down migrate discover-products crawl crawl-catalogs crawl-faq crawl-linked-resources normalize normalize-catalogs normalize-faq normalize-linked-resources chunk chunk-catalogs chunk-faq chunk-linked-resources merge-chunks index reindex

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

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

eval-guardrails:
	python -m packages.evals.refusal_eval --golden data/golden/guardrail_golden.jsonl

verify-runtime:
	python -m packages.data_pipeline.cli verify-runtime

verify-runtime-external:
	python -m packages.data_pipeline.cli verify-runtime --check-external

docker-up:
	docker compose up --build

docker-down:
	docker compose down

migrate:
	alembic -c apps/api/alembic.ini upgrade head

discover-products:
	python -m packages.data_pipeline.cli discover-products

crawl:
	python -m packages.data_pipeline.cli crawl

crawl-catalogs:
	python -m packages.data_pipeline.cli crawl-catalogs

crawl-faq:
	python -m packages.data_pipeline.cli crawl-faq

crawl-linked-resources:
	python -m packages.data_pipeline.cli crawl-linked-resources

normalize:
	python -m packages.data_pipeline.cli normalize

normalize-catalogs:
	python -m packages.data_pipeline.cli normalize --input-path data/raw/vietcombank_product_catalogs_raw.jsonl --output data/normalized/vietcombank_product_catalogs_normalized.jsonl

normalize-faq:
	python -m packages.data_pipeline.cli normalize --input-path data/raw/vietcombank_faq_raw.jsonl --output data/normalized/vietcombank_faq_normalized.jsonl

normalize-linked-resources:
	python -m packages.data_pipeline.cli normalize --input-path data/raw/vietcombank_linked_resources_raw.jsonl --output data/normalized/vietcombank_linked_resources_normalized.jsonl

chunk:
	python -m packages.data_pipeline.cli chunk

chunk-catalogs:
	python -m packages.data_pipeline.cli chunk --input-path data/normalized/vietcombank_product_catalogs_normalized.jsonl --output data/chunks/vietcombank_product_catalogs_chunks.jsonl --max-chars 4000 --overlap 0

chunk-faq:
	python -m packages.data_pipeline.cli chunk --input-path data/normalized/vietcombank_faq_normalized.jsonl --output data/chunks/vietcombank_faq_chunks.jsonl

chunk-linked-resources:
	python -m packages.data_pipeline.cli chunk --input-path data/normalized/vietcombank_linked_resources_normalized.jsonl --output data/chunks/vietcombank_linked_resources_chunks.jsonl

merge-chunks:
	python -m packages.data_pipeline.cli merge-chunks --include-product-catalogs

index:
	python -m packages.data_pipeline.cli index

reindex:
	python -m packages.data_pipeline.cli index --recreate

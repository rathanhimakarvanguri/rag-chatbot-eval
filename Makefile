.PHONY: install test lint eval eval-llm api ui docker

install:
	pip install -e ".[dev,ui]"

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

eval:
	python -m ragkit.eval.run

# Needs LLM_BASE_URL / LLM_API_KEY / LLM_MODEL (see .env.example)
eval-llm:
	python -m ragkit.eval.run --generator llm --judge --out reports/llm

api:
	uvicorn app.api:app --reload

ui:
	streamlit run app/ui.py

docker:
	docker build -t rag-chatbot-eval . && docker run --rm -p 8000:8000 rag-chatbot-eval

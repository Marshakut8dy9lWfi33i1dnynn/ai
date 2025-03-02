all: lint mypy test

lint:
	pylint mnemo_agentic || true
	flake8 mnemo_agentic tests || true
mypy:
	mypy mnemo_agentic || true

test:
	python -m unittest discover -s tests

.PHONY: all lint mypy test

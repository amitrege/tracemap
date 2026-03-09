PYTHON ?= python3

.PHONY: install install-demo lint test compile app

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-demo:
	$(PYTHON) -m pip install -e ".[dev,demo]"

lint:
	$(PYTHON) -m ruff check src tests app

test:
	$(PYTHON) -m pytest

compile:
	$(PYTHON) -m compileall src tests app

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py

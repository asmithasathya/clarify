PYTHON ?= python

.PHONY: install install-dev install-vllm test build-index demo eval

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

install-vllm:
	$(PYTHON) -m pip install -e ".[dev,vllm,peft]"

test:
	$(PYTHON) -m pytest

build-index:
	$(PYTHON) scripts/build_statute_index.py --config configs/default.yaml

demo:
	$(PYTHON) scripts/demo_example.py

eval:
	$(PYTHON) scripts/run_eval.py --config configs/default.yaml --methods closed_book rag_direct hedge revise_verify


PYTHON ?= python

.PHONY: install install-dev install-vllm test demo eval download

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

install-vllm:
	$(PYTHON) -m pip install -e ".[dev,vllm,peft]"

test:
	$(PYTHON) -m pytest

download:
	$(PYTHON) -m scripts.download_infoquest

demo:
	$(PYTHON) -m scripts.demo_example

eval:
	$(PYTHON) scripts/run_eval.py --methods direct_answer generic_hedge generic_clarify targeted_clarify

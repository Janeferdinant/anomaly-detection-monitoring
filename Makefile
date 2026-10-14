PY ?= python

.PHONY: install lint typecheck test test-all data eval-uci eval-smd demo all
install:
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps
lint:
	ruff check .
typecheck:
	mypy
test:            ## fast tests (what CI runs)
	pytest -q -m "not slow"
test-all:
	pytest -q
data:            ## download UCI power data and the Server Machine Dataset (checksummed)
	$(PY) scripts/download_data.py
	$(PY) scripts/download_smd.py
eval-uci:        ## original UCI evaluation (injected anomalies), ~8 min
	$(PY) scripts/run_evaluation.py
eval-benchmark:  ## new multi-method benchmark on UCI (injected) and SMD (real labels)
	$(PY) scripts/run_benchmark.py --dataset uci
	$(PY) scripts/run_benchmark.py --dataset smd
demo:
	$(PY) scripts/build_demo.py
all: lint typecheck test

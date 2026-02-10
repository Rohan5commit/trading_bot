PYTHON ?= python3

.PHONY: lint test backtest report run_eod run_signals run_scheduler

lint:
	$(PYTHON) -m ruff check src tests

test:
	$(PYTHON) -m pytest -q

backtest:
	$(PYTHON) -m src.main backtest --config config.yaml

report:
	$(PYTHON) -m src.main report --config config.yaml

run_eod:
	$(PYTHON) -m src.main run_eod --config config.yaml

run_signals:
	$(PYTHON) -m src.main run_signals --config config.yaml

run_scheduler:
	$(PYTHON) -m src.main run_scheduler --config config.yaml

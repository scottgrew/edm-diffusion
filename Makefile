PYTHON ?= python3
TOOLS ?= $(HOME)/venvs/tracker/bin
SOURCES := toy tests train.py figure.py

.PHONY: test check figure
test: ## run the exact-answer tests
	$(PYTHON) -m pytest -q

check: ## black and ruff (settings in pyproject.toml), 80 columns
	$(TOOLS)/black --check --quiet $(SOURCES)
	$(TOOLS)/ruff check $(SOURCES)

figure: ## docs/figure.png: data, trained sampler, guided samples + predictions
	$(PYTHON) figure.py

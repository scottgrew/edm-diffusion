PYTHON ?= python3
TOOLS ?= $(HOME)/venvs/tracker/bin
SOURCES := toy tests

.PHONY: test check
test: ## run the exact-answer tests
	$(PYTHON) -m pytest -q

check: ## black and ruff (settings in pyproject.toml), 80 columns
	$(TOOLS)/black --check --quiet $(SOURCES)
	$(TOOLS)/ruff check $(SOURCES)

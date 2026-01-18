PYTHON_INTERPRETER = python3.12
VENV_NAME = marl-exp
VENV_PATH = ./$(VENV_NAME)
VENV_BIN = $(VENV_PATH)/bin
PYTHON = $(VENV_BIN)/python
PIP = $(VENV_BIN)/pip

.PHONY: setup-python
setup-python:
	@if ! command -v $(PYTHON_INTERPRETER) >/dev/null 2>&1; then \
		echo "ERROR: $(PYTHON_INTERPRETER) not found"; \
		echo "Please install Python 3.12:"; \
		echo "  brew install python@3.12"; \
		exit 1; \
	fi
	@if [ ! -d "$(VENV_PATH)" ]; then \
		echo "Creating virtual environment $(VENV_NAME)..."; \
		$(PYTHON_INTERPRETER) -m venv $(VENV_PATH); \
	else \
		echo "Virtual environment $(VENV_NAME) already exists"; \
	fi
	@echo "Installing requirements..."
	@$(PIP) install -qU pip
	@$(PIP) install -qr requirements.txt
	@echo "✓ Python environment setup complete"

.PHONY: clearml-init
clearml-init:
	@echo "Initializing ClearML..."
	@if [ -f .env ]; then \
		. .env && $(VENV_BIN)/clearml-init; \
	else \
		$(VENV_BIN)/clearml-init; \
	fi
	@echo "✓ ClearML initialized"

.PHONY: dynare-models
dynare-models:
	@echo "Generating dynare models..."
	@$(PYTHON) lib/dynare_traj2rl_transitions.py metadata.num_samples=1 $(ARGS)
	@echo "✓ Dynare models generation complete"

.PHONY: dataset
dataset: dynare-models
	@echo "Generating dataset..."
	@$(PYTHON) lib/generate_dataset.py \
		train.dynare_output_path=data \
		val.dynare_output_path=data \
		workdir=data/processed $(ARGS)
	@echo "✓ Dataset generation complete"

.PHONY: pipeline-exp
pipeline-exp:
	@echo "Running pipeline (experimental setup)..."
	@$(PYTHON) pipeline/run_pipeline.py \
		metadata.output_dir=experiments \
		metadata.track=False \
		dataset.enabled=False \
		dataset.workdir=data/processed \
		train.epochs=1 \
		dataset.train.dynare_output_path=data \
		dataset.val.dynare_output_path=data \
		metadata.comment="experimental run" \
		$(ARGS)
	@echo "✓ Experimental pipeline complete"

## Run Pipeline (Production Setup)
.PHONY: pipeline-prod
pipeline-prod: setup-python clearml-init
	@echo "Running pipeline (production setup)..."
	@$(PYTHON) pipeline/run_pipeline.py \
		metadata.output_dir=marl_experiments \
		metadata.track=True \
		metadata.comment="Production run" \
		$(ARGS)
	@echo "✓ Production pipeline complete"

.PHONY: venv-activate
venv-activate:
	@if [ ! -d "$(VENV_PATH)" ]; then \
		echo "ERROR: Virtual environment not found. Run 'make setup-python' first"; \
		exit 1; \
	fi
	@echo "To activate the virtual environment, run:"
	@echo "  source $(VENV_PATH)/bin/activate"

.PHONY: venv-deactivate
venv-deactivate:
	@echo "To deactivate the virtual environment, run:"
	@echo "  deactivate"

## Run Tests
.PHONY: test
test: setup-python
	@$(PYTHON) tests/test_envs.py -vvv -s

## Build Julia image with Dynare
.PHONY: build-dynare
build-dynare:
	@echo "Building Julia Dynare Docker image..."
	@docker build -f ./dynare/docker/julia.Dockerfile -t julia-dynare .
	@echo "✓ Docker image built"

## Clean virtual environment
.PHONY: clean
clean:
	@echo "Removing virtual environment..."
	@rm -rf $(VENV_PATH)
	@echo "✓ Virtual environment removed"

## Clean all generated files
.PHONY: clean-all
clean-all: clean
	@echo "Cleaning generated files..."
	@rm -rf experiments marl_experiments models/*
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "✓ All generated files cleaned"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available commands:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Setup Working Environment
.PHONY: setup
setup:
	$(PYTHON_INTERPRETER) -m pip install -qU pip
	$(PYTHON_INTERPRETER) -m pip install -qr requirements.txt
	@pre-commit install

## Run Experiment Pipeline
.PHONY: pipeline
pipeline:
	$(PYTHON_INTERPRETER) pipeline/run_pipeline.py \
  	metadata.output_dir=marl_experiments \
  	metadata.track=True



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

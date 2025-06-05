#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Setup Working Environment
.PHONY: setup
setup:
	$(PYTHON_INTERPRETER) -m pip install -qU pip
	$(PYTHON_INTERPRETER) -m pip install -qr requirements.txt
	@pre-commit install
	@. .env && clearml-init

## Run Experiment Pipeline
.PHONY: pipeline
pipeline: setup
	@$(PYTHON_INTERPRETER) pipeline/run_pipeline.py \
  	metadata.output_dir=marl_experiments \
  	metadata.track=True

## Build Julia image with Dynare and some other packages
.PHONY: build_julia_dynare_image
build_julia_dynare_image:
	docker build -f ./dynare/docker/julia.Dockerfile -t julia-dynare .

## Run Dynare models simulations
.PHONY: dynare
dynare:
	@$(PYTHON_INTERPRETER) lib/dynare_traj2rl_transitions.py $(ARGS)

test:
	@$(PYTHON_INTERPRETER) tests/test_envs.py

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

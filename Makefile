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


## Run python script for translation dynare simulations
## to RL transitions
.PHONY: dynare2rl
dynare2rl:
	@$(PYTHON_INTERPRETER) lib/dynare_traj2rl_transitions.py

## Run Dynare models simulations
.PHONY: dynare
dynare:
	docker run -it \
	-v ./dynare/docker/dynare_models:/app/input \
	-v ./data/raw/:/app/output \
	-v ./dynare/docker/main.jl:/app/main.jl \
	-v ./dynare/conf/config.yaml:/app/config.yaml \
	julia-dynare julia main.jl --input_dir input --output_dir output --config_path config.yaml
	@$(PYTHON_INTERPRETER) lib/dynare_traj2rl_transitions.py


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

# Useful Commands
1. Start with `make setup-python`. After this step you will have a virtual envirnoment with all packages. 
2. Install octave + dynare (`brew install dynare`) and install in octave required packages `pkg install -forge io statistics`.
3. Set correct `DYNARE_PATH=/opt/homebrew/opt/dynare/lib/dynare/matlab` in env.
4. Try to build exp dataset: `make dataset`
5. To run test pipeline: `make pipeline-exp`
6*. To generate data for a specific environment run python command (an example):\
```python lib/data/generate_data.py --env_class lib.envs.rbc.RBCEnv --num_steps 1000 --discount_rate 0.98```

## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make setup`.
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- Proccessed tracks; ready for RL experiment.
│   ├── interim        <- Those tracks that were matched with Dynare and the logs
│   │                     for those that were not matched.
│   └── raw            <- Raw generated tracks.
│
├── examples           <- Jupyter notebooks guides.
│
├── pyproject.toml     <- Project configuration file with package metadata for lib.
│
├── docs               <- Documentation.
│
├── lib
│   ├── __init__.py             <- Makes lib a Python module
│   ├── config.py               <- Store useful variables and configuration
│   ├── dataset.py              <- Scripts to generate raw tracks
│   ├── dataset_process.py      <- Code to prepare tracks for RL algo training
│   ├── utility_funcs.py        <- Store economical utility functions
│   ├── validation.py           <- Code to compare RL environments to Dynare
│   ├── train.py                <- Code to train a model
│   ├── predict.py              <- Code to run trained model on a test set
│   ├── models                  <- ML-models Classes
│   └── plots.py                <- Code to create visualizations
│
├── conf                        <- Hydra configuration files.
│   ├── config.yaml             <- Main configuration file with the main hyperparameters.
│   └── env                     <- Directory to store environment-specific configuration files.
│       ├── rbc.yaml            <- RBCEnv configuration file.
│       ├── ncg.yaml            <- NCGEnv configuration file.
│       └── marl_macroeconomics.yaml <- MARLMacroeconomicEnv configuration file.
│
└── clearml.conf                <- clearml configuration file with credentials
```

--------
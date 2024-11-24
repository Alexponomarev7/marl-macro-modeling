# Experiment Pipeline
1. Create environments
2. Solve them analytically
3. Save tracks from (2) to the dataset
4. Train Transformer model

# Useful Commands
1. Setup working env: `make setup`
2. To generate data run python command (an example):\
```python lib/data/generate_data.py --env_class lib.envs.rbc.RBCEnv --num_steps 1000 --discount_rate 0.98```

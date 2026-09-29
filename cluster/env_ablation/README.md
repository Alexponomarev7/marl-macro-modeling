# Env-count ablation on a023-mts

One job on one A100 does two things at once:

- **CPUs** generate the Dynare data one env at a time (Octave + Dynare).
- **GPU** trains on the data that already exists. Up to 3 runs share the card.

Generation and training are two independent processes. Nothing on the training side can stop the generation:

- **Separate process.** Generation runs as its own process (`run.py generate`, in its own session, logging to `logs/generate.log`). It only writes data and `status/*.json` markers.
- **Training only reads markers.** The scheduler polls the markers and starts runs as envs become ready. A training run that crashes or fails is only logged. The scheduler catches its own errors, because it must stay alive for as long as generation runs: the job ending would stop the pod.
- **Generation has CPU priority.** Dynare runs at normal priority and the training runs are `nice 10`.
- **Failures inside generation.** Each Dynare stage is tried twice. An env that still fails is marked skipped and generation moves on to the next one. If the generation process itself dies, it is restarted, up to 5 times, and resumes from the draws already written.

## Design

**Test set (fixed).** Two whole families are held out, so no env of theirs is in any training set:

- open-economy households: Aguiar-Gopinath, García-Cicco, SGU 2003, SGU 2004
- non-household agents: Jermann-Quadrini, Kiyotaki-Moore, OLG, RBC_state_dependent_GIRF_government

The test draws are fresh (200 per env).

**Training envs.** The other 27 models (central banks, RBC households with labor, consumption-only Ramsey/RBC) are shuffled once with `ORDER_SEED=0`. The order is saved to `order.json`.

**Steps.** Step *k* trains from scratch on the first *k* envs plus the 400 procedural LQ tasks. Step 0 uses the LQ tasks alone.

- Every step gets the same budget: 100k gradient steps at batch size 32, with the model and optimizer from `pipeline/configs/train/default.yaml`.
- Each step runs with 3 seeds.

**Scores.** Each run is scored with `lib.evaluation` as NMSE(model) / NMSE("repeat the last action"), as a geomean over envs:

- `test`: held-out families, with context diagnostics and bootstrap CIs
- `test_observables`: held-out families, shocks/TFP hidden
- `seen_val`: fresh draws of the envs it trained on

Family groupings come from `reports/plan.md` (to-do 8) and live in `config.py`.

## Once: the environment (~30-40 min)

1. Put the repo at `~/kovalenko/projects/marl-macro-modeling`, either with `git clone` of the branch or a tarball.
2. In the JS terminal:

```bash
cd ~/kovalenko/projects/marl-macro-modeling
bash cluster/env_ablation/setup_env.sh      # creates /home/jovyan/kovalenko/envs/marl-dynare (~8 GB)
```

The script needs no root. It uses micromamba and conda-forge to install Python 3.11, Octave 10 and the compilers, builds Dynare 7.1 from source, installs the Octave Forge packages Dynare needs (datatypes 1.2.0, statistics 1.8.2: the last releases for Octave 10), installs CUDA torch, and ends with a smoke test. It should print `OK: 4/4 Dynare episodes` (3/4 is also accepted).

Re-running it skips the parts that are already done. If the JS terminal has no internet, build the environment in a job instead:

```bash
SETUP=1 python3.11 cluster/env_ablation/submit.py
```

## Run

```bash
cd ~/kovalenko/projects/marl-macro-modeling
DRY_RUN=1 python3.11 cluster/env_ablation/submit.py   # prints the job, submits nothing
SMOKE=1   python3.11 cluster/env_ablation/submit.py   # ~1 h: 2 envs x 40 draws, 3k steps, 2 seeds
python3.11 cluster/env_ablation/submit.py             # full run
```

After the smoke run, check that `.../env_ablation_v1_smoke/results.csv` has 6 rows (steps 0-2 × 2 seeds). Then submit the full run.

- **Output** goes to `/mnt/shared_ru.ml.SZ-2_000176/kovalenko/marl/env_ablation_v1/`. That is the 1 TB volume, and the job needs about 10-20 GB of it.
- **Code** is frozen at submit time to `~/kovalenko/runs/marl_env_ablation_v1/code`, so editing the repo later doesn't affect a queued or running job (footgun #10).
- **Resume.** Preemptions and restarts resume automatically: generated envs and finished runs leave markers. Re-submitting the same command also resumes, so after fixing something, just submit again.
- **Knobs.** Any value in `config.py` can be overridden by setting it in the environment at submit time, e.g. `SEEDS=5 MAX_STEPS=50000 python3.11 cluster/env_ablation/submit.py`. Also: `INSTANCE_TYPE`, `RUN_NAME`, `OUT_ROOT`, `ENV_PREFIX`.

## Duration (full run, `a100.1gpu.8C.243G`)

| | |
|---|---|
| Generation | ~134k training draws + ~3k val/test draws at ~2-3 s per draw per CPU core. That's ~12-15 h on 8 CPUs and scales down linearly with more CPUs. |
| Training | 28 steps × 3 seeds = 84 runs of 100k steps, 3 at a time. The first smoke run shows the real steps/s in `runs/*/train.log`. |

**Use more CPUs if you can.** Generation is CPU-bound, and the data loaders share the same CPUs. If the platform has a 1-GPU A100 instance with more CPUs, use it: `INSTANCE_TYPE=<name> python3.11 cluster/env_ablation/submit.py`. To list the available instances:

```bash
python3.11 -c "import client_lib; client_lib.get_instance_types(regions=['SR006'])"
```

## Monitor

```bash
OUT=/mnt/shared_ru.ml.SZ-2_000176/kovalenko/marl/env_ablation_v1
tail -f $OUT/logs/run.log                        # status every 30 s: envs generated, runs done/running
tail -f $OUT/logs/generate.log                   # the generation process (Dynare output: logs/dynare_*.log)
tail -f $OUT/runs/step03_seed0/train.log         # one run: steps/s, losses, scores
~/kovalenko/envs/marl-dynare/env/bin/python ~/kovalenko/runs/marl_env_ablation_v1/code/cluster/env_ablation/summarize.py $OUT
```

`summarize.py` can be run at any time. It writes:

- `results.csv`: one row per run
- `results_by_step.csv`: mean/std over seeds
- `ratio_by_env.csv`
- `curve.png`: test error against the number of training envs

The job runs it once more at the end.

## If something fails

| Where it failed | Where to look |
|---|---|
| Whole job | `~/.mlspace-logs/<job>/retry.000/rank.0/stdout` (see the operator guide) |
| One run | `$OUT/runs/<run>/train.log`. A run is retried once (`RUN_ATTEMPTS=2`). After that the job still finishes, exits 1, and lists the failed runs. |
| One Dynare env | `$OUT/logs/dynare_train_<env>.log`. Failed parameter draws are normal: they're resampled. An env with no usable draws at all is skipped in the order, and `run.log` says so. |

## Files

| File | What it does |
|---|---|
| `run.py` | orchestrator that runs inside the job |
| `train_step.py` | one training + evaluation run |
| `config.py` | families, test set, knobs |
| `packing.py` | per-env episodes packed into memory-mapped arrays, so training doesn't open one parquet per sample on NFS |
| `summarize.py` | builds the summary tables and curve |
| `submit.py` | client_lib submit |
| `job.sh` | job entrypoint |
| `setup_env.sh` | builds the environment |

"""Submit the env ablation as one job on Cloud.ru ML Space (a023-mts, SR006), from the JS terminal.

    DRY_RUN=1 python3.11 cluster/env_ablation/submit.py    # print the job, submit nothing
    SMOKE=1 python3.11 cluster/env_ablation/submit.py      # ~1 h end-to-end check, separate output dir
    python3.11 cluster/env_ablation/submit.py              # the full ablation
    SETUP=1 python3.11 cluster/env_ablation/submit.py      # build the environment in a job (if the JS has no internet)

Re-submitting the same RUN_NAME resumes: finished envs and runs are skipped. The code is frozen into
CODE_DIR at submit time, so editing the repo afterwards does not change a queued or running job.
Knobs of config.py (NUM_SAMPLES, SEEDS, MAX_STEPS, ...) set in the environment are passed to the job.
"""
import inspect
import os
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[2]
USER_TAG = os.environ.get("USER_TAG", "kovalenko")
SMOKE = os.environ.get("SMOKE") == "1"
SETUP = os.environ.get("SETUP") == "1"
DRY_RUN = os.environ.get("DRY_RUN") == "1"
RUN_NAME = os.environ.get("RUN_NAME", "env_ablation_v1") + ("_smoke" if SMOKE else "")

INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "a100.1gpu.8C.243G")
BASE_IMAGE = os.environ.get("BASE_IMAGE", "cr.ai.cloud.ru/aicloud-base-images/cuda12.3-torch2-py310:0.0.37")
ENV_PREFIX = os.environ.get("ENV_PREFIX", f"/home/jovyan/{USER_TAG}/envs/marl-dynare")
OUT_ROOT = os.environ.get("OUT_ROOT", f"/mnt/shared_ru.ml.SZ-2_000176/{USER_TAG}/marl/{RUN_NAME}")
CODE_DIR = Path(os.environ.get("CODE_DIR", f"/home/jovyan/{USER_TAG}/runs/marl_{RUN_NAME}/code"))
CHECKPOINT_DIR = f"/home/jovyan/{USER_TAG}/checkpoints/marl_{RUN_NAME}"  # platform marker; must be under /home/jovyan

KNOBS = ["NUM_SAMPLES", "VAL_SAMPLES", "TEST_SAMPLES", "LQ_EPISODES", "LQ_REGIMES", "LQ_VAL_EPISODES", "DATA_SEED",
         "ORDER_SEED", "MAX_ENVS", "SEEDS", "MAX_STEPS", "BATCH_SIZE", "VAL_EVERY", "LOADER_WORKERS",
         "MAX_PARALLEL_RUNS", "RUN_ATTEMPTS", "EVAL_WINDOWS", "EVAL_BOOTSTRAP", "LR", "DYNARE_WORKERS"]
SMOKE_KNOBS = {"NUM_SAMPLES": "40", "VAL_SAMPLES": "8", "TEST_SAMPLES": "10", "MAX_ENVS": "2", "SEEDS": "2",
               "MAX_STEPS": "3000", "VAL_EVERY": "1000", "EVAL_WINDOWS": "2", "EVAL_BOOTSTRAP": "20"}
FROZEN = ["lib", "pipeline", "research", "cluster", "dynare/conf", "dynare/docker/dynare_models", "pyproject.toml"]


def freeze_code() -> Path:
    """Copies the code the job needs (no data, no git) to CODE_DIR."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints", "data", "*.ipynb")
    for rel in FROZEN:
        src, dst = REPO / rel, CODE_DIR / rel
        if src.is_dir():
            shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return CODE_DIR


def main() -> None:
    env_variables = {"ENV_PREFIX": ENV_PREFIX, "OUT_ROOT": OUT_ROOT, "SETUP": "1" if SETUP else "0"}
    if SMOKE:
        env_variables |= SMOKE_KNOBS
    env_variables |= {k: os.environ[k] for k in KNOBS if k in os.environ}
    # values reach the pod through an unquoted `export`: keep them to numbers and plain paths
    bad = {k: v for k, v in env_variables.items() if not all(c.isalnum() or c in "/._-" for c in v)}
    if bad:
        raise ValueError(f"environment values with characters the platform's export mangles: {bad}")

    code = CODE_DIR if DRY_RUN else freeze_code()
    job_desc = f"{USER_TAG}-marl-{'setup' if SETUP else RUN_NAME}"
    kwargs = dict(
        base_image=BASE_IMAGE,
        script=f"bash {code}/cluster/env_ablation/job.sh",
        instance_type=INSTANCE_TYPE,
        n_workers=1,
        processes_per_worker=1,
        env_variables=env_variables,
        job_desc=job_desc,
        checkpoint_dir=CHECKPOINT_DIR,
        max_retry=3 if (SMOKE or SETUP) else 10,
        internet=True if SETUP else None,
    )
    print(f"job {job_desc}\n  code     {code}\n  output   {OUT_ROOT}\n  instance {INSTANCE_TYPE}")
    for k, v in kwargs.items():
        print(f"  {k} = {v}")
    if DRY_RUN:
        print("DRY_RUN: nothing submitted")
        return

    import client_lib

    kwargs |= {"type": client_lib.Job.Type.pytorch2, "region": client_lib.RegionMT.SR006}
    try:
        kwargs["health_params"] = client_lib.JobHealthParams(
            log_period=60, action=client_lib.JobHealthAction.restart, sub_actions=[client_lib.JobHealthSubAction.notify])
    except AttributeError:
        pass
    accepted = inspect.signature(client_lib.Job.__init__).parameters
    dropped = sorted(set(kwargs) - set(accepted))
    if dropped:
        print(f"  (client_lib.Job does not take {dropped}; dropped)")
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    job = client_lib.Job(**{k: v for k, v in kwargs.items() if k in accepted and v is not None})
    result = job.submit()
    print(f"submitted: {result}")
    print(f"logs: tail -f {OUT_ROOT}/logs/run.log   (worker stdout: ~/.mlspace-logs/<job>/retry.000/rank.0/stdout)")


if __name__ == "__main__":
    main()

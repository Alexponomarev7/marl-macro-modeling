"""Env-count ablation: generate the Dynare data env by env on the CPUs while the GPU trains on what is
already there.

Step k trains from scratch on the first k training envs (+ the procedural LQ tasks) and is scored on the
test envs, whose families are absent from every training set (config.py). Step 0 is the LQ tasks alone.
Each step runs SEEDS times, up to MAX_PARALLEL_RUNS runs sharing the GPU.

    OUT_ROOT=/abs/output/dir python cluster/env_ablation/run.py

Everything is resumable: finished generation stages and runs leave markers under OUT_ROOT, so
restarting the same command (e.g. after a preemption) continues where it stopped.

OUT_ROOT/
  order.json                   training envs in the order they are added
  data/train/<env>/            Dynare draws of each training env (raw/, interim/)
  data/val/, data/test/        fresh draws: validation of the training envs; test envs
  data/lq/{train,val}/         procedural LQ tasks
  packed/{train,val}/<env>/    the same episodes packed for training (packing.py)
  status/*.json                one per finished generation stage
  runs/step<k>_seed<s>/        spec.json, train.log, ckpt/, test*.csv, result.json
  results.csv, curve.png       summary (summarize.py)
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]

from loguru import logger
from omegaconf import OmegaConf

import config as C
from lib.lq_tasks import write_tasks
from packing import pack_episodes

OUT = Path(os.environ["OUT_ROOT"]).resolve()
LQ = "LQ_random"


def available_cpus() -> int:
    """CPUs this container may use: the cgroup quota if there is one, else the affinity mask."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1


def done(stage: str) -> dict | None:
    marker = OUT / "status" / f"{stage}.json"
    return json.loads(marker.read_text()) if marker.exists() else None


def mark(stage: str, **info) -> None:
    marker = OUT / "status" / f"{stage}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker.with_suffix(".tmp")
    tmp.write_text(json.dumps(info, indent=2))
    os.replace(tmp, marker)


def episodes_of(model: str, interim: Path) -> list[Path]:
    return [f for f in interim.glob(f"{model}_config_*.parquet") if f.name.rsplit("_config_", 1)[0] == model]


def dynare(models: list[str], folder: Path, samples: int, seed: int, tag: str) -> None:
    """lib/dynare_traj2rl_transitions.py for `models`: draws into folder/raw, episodes into folder/interim."""
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "lib" / "dynare_traj2rl_transitions.py"),
        f"metadata.data_folder={folder}", f"metadata.num_samples={samples}", f"metadata.seed={seed}",
        "metadata.resume=true", f"metadata.only_models=[{','.join(models)}]", f"hydra.run.dir={OUT / 'logs' / 'hydra' / tag}",
    ]
    env = os.environ | {"DYNARE_WORKERS": str(DYNARE_WORKERS), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
    started = time.monotonic()
    with open(OUT / "logs" / f"dynare_{tag}.log", "a") as log:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    logger.info(f"dynare {tag}: {len(models)} models in {(time.monotonic() - started) / 60:.1f} min")


def attempt(stage: str, fn) -> bool:
    """fn() up to GEN_ATTEMPTS times (Dynare resumes from the draws already written); a stage that keeps
    failing is logged and left to the caller, so one bad env never stops the generation."""
    for k in range(1, GEN_ATTEMPTS + 1):
        try:
            fn()
            return True
        except Exception as e:
            logger.opt(exception=e).error(f"generation of {stage} failed (attempt {k}/{GEN_ATTEMPTS})")
    return False


def generate(order: list[str]) -> None:
    """LQ tasks, then the test envs, then the training envs one by one. Independent of the training runs:
    it only writes data and markers, which the scheduler polls."""
    lq = {"periods": 1000, "regimes": C.LQ_REGIMES}

    def lq_tasks():
        for split, episodes, seed in (("train", C.LQ_EPISODES, C.DATA_SEED), ("val", C.LQ_VAL_EPISODES, C.DATA_SEED + 1)):
            files = write_tasks(OUT / "data" / "lq" / split, episodes, seed=seed, **lq) if episodes else []
            pack_episodes(files, OUT / "packed" / split / LQ)
        mark("lq", episodes=C.LQ_EPISODES)

    if not done("lq") and not attempt("lq", lq_tasks):
        raise RuntimeError("the LQ tasks could not be generated")

    if not done("test"):
        ok = attempt("test", lambda: dynare(C.TEST_ENVS, OUT / "data" / "test", C.TEST_SAMPLES, C.DATA_SEED + 2, "test"))
        interim = OUT / "data" / "test" / "interim"
        # scored on whatever test draws exist, so that the runs never wait forever
        mark("test", episodes={m: len(episodes_of(m, interim)) for m in C.TEST_ENVS}, complete=ok)
        logger.info(f"test data: {done('test')['episodes']}")

    for model in order:
        if done(f"train_{model}"):
            continue
        started = time.monotonic()
        folder, val = OUT / "data" / "train" / model, OUT / "data" / "val"
        counts = {}

        def env_data():
            dynare([model], folder, C.NUM_SAMPLES, C.DATA_SEED, f"train_{model}")
            counts["episodes"] = pack_episodes(episodes_of(model, folder / "interim"), OUT / "packed" / "train" / model)
            dynare([model], val, C.VAL_SAMPLES, C.DATA_SEED + 1, f"val_{model}")
            counts["val_episodes"] = pack_episodes(episodes_of(model, val / "interim"), OUT / "packed" / "val" / model)

        if not attempt(model, env_data):
            counts = {"episodes": 0, "val_episodes": 0, "error": f"failed, see logs/dynare_*_{model}.log"}
        mark(f"train_{model}", **counts, minutes=(time.monotonic() - started) / 60)
        n, n_val = counts["episodes"], counts["val_episodes"]
        logger.info(f"env {model}: {n} train / {n_val} val episodes" + (" - no usable draws, skipped" if not (n and n_val) else ""))


def steps(order: list[str]) -> list[list[str]]:
    """Training envs of every step that can run now: the first k usable envs of the order, for as long as
    the order's prefix is generated."""
    if not done("lq"):
        return []
    usable, out = [], [[]] if C.LQ_EPISODES else []
    for model in order:
        status = done(f"train_{model}")
        if status is None:
            break
        if status["episodes"] and status["val_episodes"]:
            usable.append(model)
            out.append(list(usable))
    return out


def spec(step: int, seed: int, envs: list[str]) -> dict:
    packed = lambda split: [str(OUT / "packed" / split / m) for m in ([LQ] if C.LQ_EPISODES else []) + envs]
    return {
        "step": step, "seed": seed, "envs": envs, "train_packed": packed("train"), "val_packed": packed("val"),
        "val_dir": str(OUT / "data" / "val" / "interim"), "test_dir": str(OUT / "data" / "test" / "interim"),
        "test_done": str(OUT / "status" / "test.json"),
        "max_steps": C.MAX_STEPS, "batch_size": C.BATCH_SIZE, "val_every": C.VAL_EVERY, "lr": C.LR,
        "loader_workers": C.LOADER_WORKERS, "eval_windows": C.EVAL_WINDOWS, "eval_bootstrap": C.EVAL_BOOTSTRAP,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    logger.add(OUT / "logs" / "run.log")
    order_file = OUT / "order.json"
    if order_file.exists():  # the order is fixed at the first start
        order = json.loads(order_file.read_text())["train_envs"]
    else:
        models = list(OmegaConf.load(ROOT / "dynare" / "conf" / "config.yaml")["models"])
        order = C.train_env_order(models)
        missing = sorted(set(C.TEST_ENVS) - set(models))
        if missing:
            raise KeyError(f"test envs not in dynare/conf/config.yaml: {missing}")
        order_file.write_text(json.dumps({"test_envs": C.TEST_ENVS, "train_envs": order}, indent=2))
    logger.info(f"{available_cpus()} CPUs, {DYNARE_WORKERS} Dynare workers | test envs: {C.TEST_ENVS}")
    logger.info(f"training envs, in order: {order}")

    def start_generation() -> subprocess.Popen:
        log = open(OUT / "logs" / "generate.log", "a")
        # its own process and session: nothing that happens to the scheduler or a training run stops it
        return subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "generate"], cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    generation, gen_starts = start_generation(), 1

    gen_failed = False
    running: dict[Path, tuple[subprocess.Popen, object]] = {}
    attempts: dict[Path, int] = {}
    last_status = 0.0
    while True:
        try:
            todo = []
            for envs in steps(order):
                step = len(envs)
                for seed in range(C.SEEDS):
                    run_dir = OUT / "runs" / f"step{step:02d}_seed{seed}"
                    if (run_dir / "result.json").exists() or run_dir in running or attempts.get(run_dir, 0) >= C.RUN_ATTEMPTS:
                        continue
                    todo.append((run_dir, spec(step, seed, envs)))
            for run_dir, run_spec in todo[:max(0, C.MAX_PARALLEL_RUNS - len(running))]:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "spec.json").write_text(json.dumps(run_spec, indent=2))
                log = open(run_dir / "train.log", "a")
                attempts[run_dir] = attempts.get(run_dir, 0) + 1
                # niced: Dynare has the CPUs first, the runs' data loaders take the rest
                running[run_dir] = (subprocess.Popen(["nice", "-n", "10", sys.executable, str(HERE / "train_step.py"), str(run_dir)],
                                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log)
                logger.info(f"started {run_dir.name} ({len(run_spec['envs'])} envs, attempt {attempts[run_dir]})")
            for run_dir, (proc, log) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close()
                del running[run_dir]
                if proc.returncode == 0:
                    result = json.loads((run_dir / "result.json").read_text())
                    logger.info(f"finished {run_dir.name}: test geomean(model/persistence) {result['test']['geomean_ratio']:.4f}")
                else:
                    logger.error(f"{run_dir.name} failed (exit {proc.returncode}), see {run_dir / 'train.log'}")
            if generation is not None and generation.poll() is not None:
                if generation.returncode == 0:
                    logger.info("generation finished")
                    generation = None
                elif gen_starts < GEN_RESTARTS:
                    logger.error(f"generation exited with {generation.returncode}, restarting it (see logs/generate.log)")
                    generation, gen_starts = start_generation(), gen_starts + 1
                else:
                    logger.error(f"generation exited with {generation.returncode} {gen_starts} times, giving up on it")
                    gen_failed, generation = True, None
            if generation is None and not running and not todo:
                break
            if time.monotonic() - last_status > 30:  # also a heartbeat for the platform's log watchdog
                n_done = len(list((OUT / "runs").glob("*/result.json")))
                envs_done = sum(done(f"train_{m}") is not None for m in order)
                logger.info(f"status: {envs_done}/{len(order)} envs generated | runs: {n_done} done, "
                            f"{len(running)} running, {len(todo)} waiting")
                last_status = time.monotonic()
        except Exception as e:  # the scheduler must outlive any bad run; the job ending would kill the generation
            logger.opt(exception=e).error("scheduler error, continuing")
        time.sleep(10)

    subprocess.run([sys.executable, str(HERE / "summarize.py"), str(OUT)], cwd=ROOT, check=False)
    given_up = sorted(d.name for d, n in attempts.items() if n >= C.RUN_ATTEMPTS and not (d / "result.json").exists())
    skipped = [m for m in order if (done(f"train_{m}") or {}).get("error")]
    if gen_failed or given_up or skipped:
        logger.error(f"finished with errors: generation {'failed' if gen_failed else 'ok'}, envs skipped after "
                     f"errors: {skipped}, failed runs: {given_up}")
        sys.exit(1)
    logger.info(f"all done: {OUT / 'results.csv'}")


DYNARE_WORKERS = int(os.environ.get("DYNARE_WORKERS", available_cpus()))
GEN_ATTEMPTS = 2  # per generation stage
GEN_RESTARTS = 5  # of the whole generation process, if it dies

if __name__ == "__main__":
    if sys.argv[1:] == ["generate"]:
        logger.add(OUT / "logs" / "run.log")
        generate(json.loads((OUT / "order.json").read_text())["train_envs"])
    else:
        main()

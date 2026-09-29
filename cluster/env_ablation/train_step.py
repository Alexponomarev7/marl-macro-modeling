"""One run of the env-count ablation: train from scratch on the envs of spec.json, then score the best
checkpoint on the held-out test envs (and on fresh draws of the envs it was trained on).

    python cluster/env_ablation/train_step.py <run_dir>     # run_dir holds spec.json, written by run.py

Resumes from <run_dir>/ckpt/last.ckpt; writes <run_dir>/result.json last.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "pipeline")]

from hydra import compose, initialize_config_dir
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from loguru import logger
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from lib.dataset import LATENT_STATES, Tokenizer
from lib.evaluation import evaluate, load_policy
from packing import PackedEconomicsDataset, index_entries
from run_pipeline import EconomicPolicyModel


def train_config(lr: float) -> dict:
    """pipeline/configs/train/default.yaml, resolved."""
    with initialize_config_dir(config_dir=str(ROOT / "pipeline" / "configs"), version_base=None):
        cfg = compose("pipeline.yaml", overrides=[
            "metadata.output_dir=unused", "metadata.comment=env_ablation", "metadata.run_id=env_ablation",
            "dataset.train.dynare_output_path=unused", "dataset.val.dynare_output_path=unused",
        ])
    train = OmegaConf.to_container(cfg, resolve=True)["train"]
    train["model"]["num_tasks"] = Tokenizer().num_tasks
    if lr:
        train["optimizer"]["lr"] = lr
    return train


def dataset(index_dir: Path, packed_dirs: list[str], cfg: dict, train: bool) -> PackedEconomicsDataset:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "metadata.json").write_text(json.dumps([e for d in packed_dirs for e in index_entries(Path(d))]))
    return PackedEconomicsDataset(
        index_dir, cfg["max_state_dim"], cfg["max_action_dim"], cfg["max_endogenous_dim"],
        cfg["max_model_params_dim"], cfg["max_seq_len"], random_window=train,
        **({k: cfg.get(k, 0.0) for k in ("state_dropout", "state_noise", "hide_latent")} if train else {}),
    )


class Progress(Callback):
    def __init__(self, every: int):
        self.every, self.started = every, time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step % self.every == 0:
            loss = trainer.callback_metrics.get("train_loss_step")
            rate = step / max(time.monotonic() - self.started, 1e-9)
            logger.info(f"step {step}/{trainer.max_steps} | train_loss {float(loss) if loss is not None else float('nan'):.4f} | {rate:.1f} steps/s")


def score(table) -> dict:
    """The headline numbers of lib.evaluation.summarize, plus each env's model/persistence ratio."""
    ratio = (table.model_nmse + 1e-6) / (table.persistence_nmse + 1e-6)
    out = {
        "envs": len(table),
        "geomean_ratio": float(np.exp(np.log(ratio).mean())),
        "median_model_nmse": float(table.model_nmse.median()),
        "median_persistence_nmse": float(table.persistence_nmse.median()),
        "median_oracle_nmse": float(table.oracle_nmse.median()),
        "better_than_persistence": int((table.model_nmse < table.persistence_nmse).sum()),
        "better_than_oracle": int((table.model_nmse < table.oracle_nmse).sum()),
        "ratio_by_env": dict(zip(table.env, map(float, ratio))),
    }
    if "shuffled_last" in table:
        out |= {k: float(table[k].median()) for k in ("model_last", "shuffled_last", "truncated_last")}
    return out


def main(run_dir: Path) -> None:
    spec = json.loads((run_dir / "spec.json").read_text())
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    L.seed_everything(spec["seed"], workers=True)
    cfg = train_config(spec["lr"])

    train_set = dataset(run_dir / "index" / "train", spec["train_packed"], cfg, train=True)
    val_set = dataset(run_dir / "index" / "val", spec["val_packed"], cfg, train=False)
    logger.info(f"step {spec['step']} seed {spec['seed']}: {len(spec['envs'])} envs, {len(train_set)} train / {len(val_set)} val episodes")
    workers = spec["loader_workers"]
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=spec["batch_size"], shuffle=shuffle, num_workers=workers,
                                            pin_memory=True, persistent_workers=workers > 0)

    model = EconomicPolicyModel(
        model_cfg=cfg["model"], optimizer_cfg=cfg["optimizer"], scheduler_cfg=cfg["scheduler"],
        criterion_cfg=cfg["loss"], state_max_dim=cfg["max_state_dim"], action_max_dim=cfg["max_action_dim"],
        endogenous_max_dim=cfg["max_endogenous_dim"], dynamics_weight=cfg.get("dynamics_weight", 0.1),
    )
    ckpt_dir = run_dir / "ckpt"
    best = ModelCheckpoint(dirpath=ckpt_dir, filename="best", monitor="val_action_loss", save_top_k=1, save_last=True)
    trainer = L.Trainer(
        max_steps=spec["max_steps"], max_epochs=-1, val_check_interval=spec["val_every"],
        check_val_every_n_epoch=None, gradient_clip_val=cfg["gradient_clip_val"], accelerator="auto", devices=1,
        callbacks=[best, Progress(every=1000)], logger=CSVLogger(run_dir, name="logs"),
        enable_progress_bar=False, enable_model_summary=False,
    )
    last = ckpt_dir / "last.ckpt"
    trainer.fit(model, loader(train_set, True), loader(val_set, False), ckpt_path=str(last) if last.exists() else None)

    checkpoint = Path(best.best_model_path) if best.best_model_path else last
    logger.info(f"scoring {checkpoint}")
    while not Path(spec["test_done"]).exists():  # test draws are generated alongside the first training envs
        logger.info("waiting for the test data")
        time.sleep(60)
    test_models = [m for m, n in json.loads(Path(spec["test_done"]).read_text())["episodes"].items() if n]
    policy = load_policy(checkpoint)
    tables = {
        "test": evaluate(policy, Path(spec["test_dir"]), spec["eval_windows"], context_diagnostics=True,
                         bootstrap=spec["eval_bootstrap"], include_models=test_models),
        "test_observables": evaluate(policy, Path(spec["test_dir"]), spec["eval_windows"],
                                     hidden_states=LATENT_STATES, include_models=test_models),
    }
    if spec["envs"]:
        tables["seen_val"] = evaluate(policy, Path(spec["val_dir"]), spec["eval_windows"], include_models=spec["envs"])
    for name, table in tables.items():
        table.to_csv(run_dir / f"{name}.csv", index=False)

    result = {
        "step": spec["step"], "seed": spec["seed"], "n_envs": len(spec["envs"]), "envs": spec["envs"],
        "train_episodes": len(train_set), "checkpoint": str(checkpoint),
        "best_val_action_loss": float(best.best_model_score) if best.best_model_score is not None else None,
        **{name: score(table) for name, table in tables.items()},
    }
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, run_dir / "result.json")
    logger.info(f"test geomean(model/persistence) {result['test']['geomean_ratio']:.4f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

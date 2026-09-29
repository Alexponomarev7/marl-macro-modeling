"""Env-count ablation: which Dynare models are held out, which are added one at a time, and the knobs.

Families follow reports/plan.md (to-do 8). The test set is two whole families, so no env of theirs is
in any training set; training envs are added one at a time, in a fixed random order.

Every numeric knob can be overridden by an environment variable of the same name.
"""
import os
import random

FAMILIES = {
    "central_banks": [
        "Born_Pfeifer_2018_MP", "Faia_2008", "Faia_Monacelli_2008", "Gali_2010", "Gali_2015_chapter_3",
        "Gali_2015_chapter_5_commitment", "Gali_2015_chapter_5_discretion", "Gali_Monacelli_2005", "Ireland_2004",
    ],
    "open_economy_households": ["Aguiar_Gopinath_2007", "GarciaCicco_2010", "SGU_2003", "SGU_2004"],
    "non_household_agents": [
        "Jermann_Quadrini_2012", "Kiyotaki_Moore_1997", "OLG", "RBC_state_dependent_GIRF_government",
    ],
    "rbc_households_with_labor": [
        "Caldara_et_al_2012", "Collard_2001", "Hansen_1985", "McCandless_2008_Chapter_9",
        "McCandless_2008_Chapter_13", "RBC_capital_stock_shock_pf", "RBC_capital_stock_shock_stoch",
        "RBC_news_shock_model_pf", "RBC_news_shock_model_stoch", "RBC_state_dependent_GIRF_household",
        "Gali_2008_chapter_2", "Jermann_1998",
    ],
    "consumption_only": [
        "Ramsey_base", "Ramsey_cara", "Ramsey_crra", "Ramsey_upgrade", "RBC_baseline_pf", "RBC_baseline_stoch",
    ],
}
TEST_FAMILIES = ["open_economy_households", "non_household_agents"]
FAMILY_OF = {model: family for family, models in FAMILIES.items() for model in models}
TEST_ENVS = [m for f in TEST_FAMILIES for m in FAMILIES[f]]


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# data: Dynare draws per model (times each model's num_samples coefficient in dynare/conf/config.yaml)
NUM_SAMPLES = _int("NUM_SAMPLES", 6000)
VAL_SAMPLES = _int("VAL_SAMPLES", 64)
TEST_SAMPLES = _int("TEST_SAMPLES", 200)
LQ_EPISODES = _int("LQ_EPISODES", 400)  # procedural LQ tasks in every training set, as in the default dataset
LQ_REGIMES = _int("LQ_REGIMES", 3)
LQ_VAL_EPISODES = _int("LQ_VAL_EPISODES", 32)
DATA_SEED = _int("DATA_SEED", 0)
ORDER_SEED = _int("ORDER_SEED", 0)  # order in which training envs are added
MAX_ENVS = _int("MAX_ENVS", 0)  # 0: all training envs; else stop after this many (smoke runs)

# training: the same number of gradient steps at every step of the ablation
SEEDS = _int("SEEDS", 3)
MAX_STEPS = _int("MAX_STEPS", 100_000)
BATCH_SIZE = _int("BATCH_SIZE", 32)
VAL_EVERY = _int("VAL_EVERY", 2_500)
LOADER_WORKERS = _int("LOADER_WORKERS", 2)
MAX_PARALLEL_RUNS = _int("MAX_PARALLEL_RUNS", 3)  # training processes sharing the GPU
RUN_ATTEMPTS = _int("RUN_ATTEMPTS", 2)
EVAL_WINDOWS = _int("EVAL_WINDOWS", 8)  # random windows per test episode
EVAL_BOOTSTRAP = _int("EVAL_BOOTSTRAP", 200)
LR = _float("LR", 0.0)  # 0: pipeline/configs/train/default.yaml


def train_env_order(dynare_models: list[str]) -> list[str]:
    """Training envs in the order they are added: every configured Dynare model outside the test families,
    shuffled with ORDER_SEED."""
    unknown = sorted(set(dynare_models) - set(FAMILY_OF))
    if unknown:
        raise KeyError(f"models in dynare/conf/config.yaml without a family in {__file__}: {unknown}")
    order = sorted(m for m in dynare_models if m not in TEST_ENVS)
    random.Random(ORDER_SEED).shuffle(order)
    return order[:MAX_ENVS] if MAX_ENVS else order

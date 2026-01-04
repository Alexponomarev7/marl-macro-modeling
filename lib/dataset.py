import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis, skew
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

STATE_MAPPING = {
    "Empty": 0,
    "Output": 1,
    "Consumption": 2,
    "Capital": 3,
    "LoggedProductivity": 4,
    "Debt": 5,
    "InterestRate": 6,
    "PreferenceShock": 7,
    "CountryPremiumShock": 8,
    "TechGrowthRate": 9,
    "MUConsumption": 10,
    # "Hours Worked": 4,
    # "Total Factor Productivity": 5,
    # "Annualized Interest Rate": 6,
    # "Real Wage": 7,
    # "Investment": 8,
    # "Technology Shock": 9,
    # "Labor": 10,
    # "Preference Shock": 11,
    # "Marginal Utility": 12,
    # "Utility": 13,
    # "Price Inflation": 14,
    # "Debt": 15,
    # "Trade Balance To Output Ratio": 16,
    # "Trade Balance to Output Ratio": 16,
    # "Output Gap": 17,
    # "Current Account To Output Ratio": 18,
    # "Natural Output": 19,
    # "Output Deviation From Steady State": 20,
    # "Bond Price": 21,
    # "Government Spending": 22,
    # "Marginal Utility of Consumption": 23,
    # "Marginal Utility of Labor": 24,
    # "Consumption to GDP Ratio": 25,
    # "Investment to GDP Ratio": 26,
    # "Net Exports": 27,
    # "Log Output": 28,
    # "Log Consumption": 29,
    # "Log Investment": 30,
    # "Output Growth": 31,
    # "Natural Interest Rate": 32,
    # "Real Interest Rate": 33,
    # "Nominal Interest Rate": 34,
    # "Real Money Stock": 35,
    # "Money Growth Annualized": 36,
    # "Nominal Money Stock": 37,
    # "AR(1) Monetary Policy Shock Process": 38,
    # "AR(1) Technology Shock Process": 39,
    # "AR(1) Preference Shock Process": 40,
    # "Price Level": 41,
    # "Nominal Wage": 42,
    # "Real Wage Gap": 43,
    # "Wage Inflation": 44,
    # "Natural Real Wage": 45,
    # "Markup": 46,
    # "Annualized Wage Inflation Rate": 47,
    # "Value Function": 48,
    # "Auxiliary Variable For Value Function": 49,
    # "Expected Stochastic Discount Factor": 50,
    # "Volatility": 51,
    # "Expected Return On Capital": 52,
    # "Risk-Free Rate": 53,
    # "Money Growth": 54,
    # "Output Growth Rate": 55,
    # "Consumption Growth Rate": 56,
    # "Investment Growth Rate": 57,
    # "Technology Growth Rate": 58,
    # "Interest Rate": 59,
    # "Country Premium Shock": 60,
    # "Productivity": 61,
    # "Real Return On Capital": 62,
    # "Real Consumption": 63,
    # "Money Stock": 64,
    # "Growth Rate Of Money Stock": 65,
    # "Foreign Price Level": 66,
    # "Foreign Bonds": 67,
    # "Foreign Interest Rate": 68,
    # "Exchange Rate": 69,
    # "Log Capital Stock": 70,
    # "Log Labor": 71,
    # "Log Real Wage": 72,
    # "Annualized Real Interest Rate": 73,
    # "Annualized Nominal Interest Rate": 74,
    # "Annualized Natural Interest Rate": 75,
    # "Annualized Inflation Rate": 76,
    # "Trade Balance": 77,
    # "Capital Stock": 78,
    # "Real Output": 79,
    # "Output Minus Consumption": 80,
    # "Lagrange Multiplier A": 81,
    # "Lagrange Multiplier B": 82,
    # "Inflation Rate": 83,
    # "Inflation": 83,
    # "Marginal Costs": 84,
    # "Market Tightness": 85,
    # "Log TFP": 86,
    # "Log Vacancies": 87,
    # "Log Wages": 88,
    # "Log Unemployment": 89,
    # "Log Tightness A": 90,
    # "Log Tightness B": 91,
    # "Vacancies": 92,
    # "Unemployment Rate": 93,
    # "Matches": 94,
    # "Meeting Rate Between Firms And Workers": 95,
    # "Employment": 96,
    # "Gross Output A": 97,
    # "Gross Output B": 98,
    # "Government Spending Shock": 99,
    # "AR(1) Technology Process": 100,
}

ACTION_MAPPING = {
    "Empty": 0,
    "Investment": 1,
    "Consumption": 2,
    "HoursWorked": 3,
}

ENV_MAPPING = {
    "Born_Pfeifer_2018_MP": 0,
    "Aguiar_Gopinath_2007": 1,
    "RBC_news_shock_model": 2,
    "Hansen_1985": 3,
    "GarciaCicco_et_al_2010": 4,
    "Caldara_et_al_2012": 5,
    "RBC_capitalstock_shock": 6,
    "SGU_2003": 7,
    "Gali_2008_chapter_2": 8,
    "Collard_2001_example1": 9,
    "McCandless_2008_Chapter_13": 10,
    "FV_et_al_2007_ABCD": 11,
    "RBC_baseline": 12,
    "RBC_state_dependent_GIRF": 13,
    "SGU_2004": 14,
    "Faia_2008": 15,
    "McCandless_2008_Chapter_9": 16,
}

def decode_env_name(env_name: str) -> int:
    prefix = env_name.rsplit('_', 1)[0]
    if prefix.endswith('_config'):
        prefix = prefix.removesuffix('_config')
    return 0
    return ENV_MAPPING[prefix]

def state_encoder(x):
    return x
def action_encoder(x):
    return x

class EconomicsDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing economic episodes data.

    This dataset handles variable-length economic episodes by loading state-action pairs
    from parquet files and preparing them for model training. The dataset performs:
    1. State-level padding/truncation to max_state_dim for uniform feature dimensions
    2. Sequence-level padding/truncation to max_seq_len for batch processing
    3. Generation of attention masks to handle variable-length sequences
    4. Task ID encoding for multi-task learning scenarios
    """

    def __init__(
        self, data_path: Path, max_state_dim: int, max_action_dim: int,
        max_endogenous_dim: int, max_model_params_dim: int, max_seq_len: int
    ):
        """
        Initialize the dataset with the given parameters.

        Args:
            data_path (Path): Path to the directory containing episode data files and metadata
            max_state_dim (int): Maximum dimension for state vectors after padding/truncation
            max_seq_len (int): Maximum sequence length for episodes (default: 512)
        """
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.max_endogenous_dim = max_endogenous_dim
        self.max_seq_len = max_seq_len
        self.max_model_params_dim = max_model_params_dim

        metadata_path = data_path / "metadata.json"
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # todo: encode with llm
        # todo: get from dataset task id
        self.task_ids = [decode_env_name(item['env_name']) for item in self.metadata]

        # todo: Encoders of environment state and action
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder

        # todo: special words embeddings
        self.sw_state_embed = [...]
        self.sw_action_embed = [...]
        self.sw_reward_embed = [...]

    def __len__(self) -> int:
        """
        Get the total number of episodes in the dataset.

        Returns:
            int: Number of episodes in the dataset.
        """
        return len(self.metadata)

    @staticmethod
    def pad_sequence(sequence: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Pad or truncate a sequence to the specified length.

        Padding is added at the beginning of the sequence, which is useful for
        maintaining recent context in time-series data.

        Args:
            sequence (torch.Tensor): Input sequence of shape (seq_len, feature_dim)
            max_len (int): Target length for the sequence

        Returns:
            torch.Tensor: Padded or truncated sequence of shape (max_len, feature_dim)
        """
        seq_len = sequence.shape[0]
        if seq_len >= max_len:
            # If sequence is longer, take the last max_len elements
            return sequence[-max_len:]
        else:
            # If sequence is shorter, pad at the beginning
            padding = torch.zeros(
                max_len - seq_len,
                *sequence.shape[1:],
                dtype=sequence.dtype,
                )
            return torch.cat([padding, sequence], dim=0)  # Padding first, then sequence

    @staticmethod
    def pad_dim(sequence: torch.Tensor, max_dim: int) -> torch.Tensor:
        """
        Pad or truncate a sequence along the feature dimension.

        Used to ensure all state vectors have the same dimensionality.

        Args:
            sequence (torch.Tensor): Input sequence of shape (seq_len, feature_dim)
            max_dim (int): Maximum feature dimension to pad/truncate to

        Returns:
            torch.Tensor: Padded or truncated sequence of shape (seq_len, max_dim)
        """
        current_dim = sequence.shape[1]
        if current_dim >= max_dim:
            return sequence[:, :max_dim]  # truncate
        else:
            padding_size = max_dim - current_dim
            padding = torch.zeros(*sequence.shape[:-1], padding_size, dtype=sequence.dtype)
            return torch.cat([sequence, padding], dim=-1)

    def __getitem__(self, idx: int):
        """
        Get a single processed episode from the dataset.

        Processing steps:
        1. Load episode data from parquet file
        2. Convert states and actions to tensors
        3. Pad states to uniform feature dimension (max_state_dim)
        4. Pad sequences to uniform length (max_seq_len)
        5. Generate attention mask for valid positions

        Args:
            idx (int): Index of the episode to retrieve
x
        Returns:
            dict: A dictionary containing:
                - states (torch.Tensor): Padded state sequences [max_seq_len, max_state_dim]
                - actions (torch.Tensor): Padded action sequences [max_seq_len, action_dim]
                - task_id (torch.Tensor): Task identifier [scalar]
                - attention_mask (torch.Tensor): Boolean mask for valid positions [max_seq_len]
        """
        data = pd.read_parquet(self.metadata[idx]["output_dir"])

        states = torch.tensor(data['state'].tolist(), dtype=torch.float32)
        endogenous = torch.tensor(data['endogenous'].tolist(), dtype=torch.float32)
        actions = torch.tensor(data['action'].tolist(), dtype=torch.float32)
        rewards = torch.tensor(data['reward'].values, dtype=torch.float32).reshape(-1, 1)
        task_id = torch.tensor(self.task_ids[idx], dtype=torch.long)

        info = data.iloc[0]["info"]
        model_params = info["model_params"]

        sorted_model_params = list(sorted(model_params.items()))
        model_params_values = torch.tensor([v for k, v in sorted_model_params] + [0] * (self.max_model_params_dim - len(sorted_model_params)), dtype=torch.float32)

        # Pad states to max_state_dim
        states = self.pad_dim(states, self.max_state_dim)
        state_description = data.iloc[0]["info"]["state_description"]
        action_description = data.iloc[0]["info"]["action_description"]
        endogenous_description = data.iloc[0]["info"]["endogenous_description"]
        states_info = torch.tensor([STATE_MAPPING[state] for state in state_description] + [0] * (self.max_state_dim - len(state_description)), dtype=torch.long)
        actions_info = torch.tensor([ACTION_MAPPING[action] for action in action_description] + [0] * (self.max_action_dim - len(action_description)), dtype=torch.long)
        endogenous_info = torch.tensor([STATE_MAPPING[endogenous] for endogenous in endogenous_description] + [0] * (self.max_endogenous_dim - len(endogenous_description)), dtype=torch.long)
        assert len(states_info) == self.max_state_dim, f"states_info length is {len(states_info)} but max_state_dim is {self.max_state_dim}"
        assert len(actions_info) == self.max_action_dim, f"actions_info length is {len(actions_info)} but max_action_dim is {self.max_action_dim}"
        # Pad actions to max_actions_dim
        actions = self.pad_dim(actions, self.max_action_dim)
        endogenous = self.pad_dim(endogenous, self.max_endogenous_dim)

        # Get original sequence length
        orig_seq_len = len(states)

        # Pad sequences to max_seq_len
        states = self.pad_sequence(states, self.max_seq_len)
        actions = self.pad_sequence(actions, self.max_seq_len)
        rewards = self.pad_sequence(rewards, self.max_seq_len)
        endogenous = self.pad_sequence(endogenous, self.max_seq_len)

        # Create attention mask
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        attention_mask[:min(orig_seq_len, self.max_seq_len)] = True

        return {
            'states': states,  # [max_seq_len, max_state_dim]
            'states_info': states_info,  # [max_state_dim]
            'actions': actions,  # [max_seq_len, action_dim]
            'actions_info': actions_info,  # [action_dim]
            'endogenous': endogenous,  # [max_seq_len, max_endogenous_dim]
            'endogenous_info': endogenous_info,  # [max_endogenous_dim]
            'reward': rewards,  # [max_seq_len, 1]
            'task_id': task_id,  # scalar
            'model_params': model_params_values,
            'attention_mask': attention_mask  # [max_seq_len]
        }


@dataclass(frozen=True)
class EnvReport:
    """Per-environment (economics model) diversity report produced by `DatasetDiversityScorer`."""

    env_name: str
    n_episodes: int
    env_names: list[str]
    state_names: list[str]
    action_names: list[str]
    endogenous_names: list[str]

    # Within-model (episode-to-episode) variety
    mean_pairwise_vacancy: float
    mean_pairwise_coverage: float
    mean_episode_embedding_knn: float

    # Cross-model (env-to-env) variety / redundancy
    nearest_env: str | None
    shared_state_frac: float | None  # |S_i ∩ S_j| / |S_i|
    intersection_over_union: float | None  # |S_i ∩ S_j| / |S_i ∪ S_j|
    intra_over_inter: float | None   # (avg intra kNN dist) / (avg inter-cluster kNN dist)


class DatasetDiversityScorer:
    """
    Diversity scorer for trajectory datasets.

    We measure data variety between episodes of the same economics model using:
      - State-action coverage: pairwise 2D coverage over (state_i, action_j) grids to avoid
        curse of dimensionality; quantile binning handles both continuous and discrete variables.
      - Avg k-NN distance between episode embeddings where each embedding consists of extracted
        time-series features for every state variable.

    To measure variety between economics models, for each model we find the nearest other model by
    shared state fraction and compute a similarity score restricted to shared states:
      - shared_state_frac = |S_i ∩ S_j| / |S_i|
      - intersection_over_union = |S_i ∩ S_j| / |S_i ∪ S_j|
      - intra_over_inter = avg_intra_cluster_kNN_dist / avg_inter_cluster_kNN_dist

    Expected dataset layout:
      dataset_path/
        metadata.json
        *.parquet (episodes)
    Each episode parquet is expected to contain:
      - 'state': array-like per timestep, shape (T, Ds)
      - 'action': array-like per timestep, shape (T, Da)
      - 'info': dict per timestep (we read the first row) with 'state_description' and
        'action_description' (recommended).
    """

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        quantile_bins: int = 10,
        knn_k: int = 5,
        cache_parquets: bool = True,
    ):
        """
        Args:
            dataset_path: Path to a dataset directory containing `metadata.json` and episode parquets.
            quantile_bins: Number of quantile bins per dimension for state-action coverage.
            knn_k: k for k-NN distance computations (within-env and cross-env).
            cache_parquets: Cache loaded parquet DataFrames in-memory (faster, more RAM).
        """
        self.dataset_path = Path(dataset_path)
        self.quantile_bins = int(quantile_bins)
        self.knn_k = int(knn_k)
        self.cache_parquets = bool(cache_parquets)

        self.metadata: list[dict[str, Any]] = json.loads((self.dataset_path / "metadata.json").read_text())

        env_to_paths: dict[str, list[Path]] = {}
        env_group_to_env_names: dict[str, set[str]] = {}
        for item in self.metadata:
            raw_env_name = str(item.get("env_name", "unknown"))
            env_group = str(item["env_group"])
            p = Path(item["output_dir"])
            env_to_paths.setdefault(env_group, []).append(p)
            env_group_to_env_names.setdefault(env_group, set()).add(raw_env_name)

        self.env_to_episode_paths = env_to_paths
        self.env_group_to_env_names = {k: sorted(list(v)) for k, v in env_group_to_env_names.items()}

        # Caches / derived state
        self._parquet_cache: dict[Path, pd.DataFrame] = {}
        self._env_state_names: dict[str, list[str]] = {}
        self._env_action_names: dict[str, list[str]] = {}
        self._env_endogenous_names: dict[str, list[str]] = {}
        self._env_episode_state_featdicts: dict[str, list[dict[str, np.ndarray]]] = {}
        self._env_episode_embeddings_full: dict[str, np.ndarray] = {}
        self._env_bin_edges: dict[str, dict[str, list[np.ndarray]]] = {}

        # Names and order of per-variable time-series features used to build embeddings.
        self.feature_names = [
            "mean",
            "std",
            "min",
            "q25",
            "median",
            "q75",
            "max",
            "skew",
            "kurtosis",
            "autocorr1",
            "trend_slope",
            "energy",
        ]

    def _calculate_trajectories_embeddings(self) -> None:
        """
        Compute per-episode embeddings for each environment.

        Embedding construction:
          - For each episode, for each state variable, extract TS features over time.
          - Concatenate features over all state variables (sorted by state name) => episode embedding.

        Side-effects:
          - Populates `_env_episode_embeddings_full` and `_env_episode_state_featdicts`.
          - Populates `_env_state_names` / `_env_action_names`.
        """
        env_featdicts: dict[str, list[dict[str, np.ndarray]]] = {}
        env_embs: dict[str, list[np.ndarray]] = {}

        for env, paths in self.env_to_episode_paths.items():
            per_episode_featdicts: list[dict[str, np.ndarray]] = []
            per_episode_embs: list[np.ndarray] = []

            state_names_ref: list[str] | None = None
            action_names_ref: list[str] | None = None
            endogenous_names_ref: list[str] | None = None

            for p in paths:
                df = self._read_parquet(p)
                state_names, action_names, endogenous_names = self._get_descriptions(df)
                if state_names_ref is None:
                    state_names_ref = list(state_names)
                if action_names_ref is None:
                    action_names_ref = list(action_names)
                if endogenous_names_ref is None:
                    endogenous_names_ref = list(endogenous_names)

                S, _A = self._get_state_action_arrays(df)

                featdict: dict[str, np.ndarray] = {}
                for j, name in enumerate(state_names):
                    featdict[str(name)] = self._extract_ts_features(S[:, j])

                per_episode_featdicts.append(featdict)

                ordered_names = sorted(featdict.keys())
                emb = np.concatenate([featdict[n] for n in ordered_names], axis=0)
                per_episode_embs.append(emb)

            self._env_state_names[env] = state_names_ref or []
            self._env_action_names[env] = action_names_ref or []
            self._env_endogenous_names[env] = endogenous_names_ref or []
            env_featdicts[env] = per_episode_featdicts
            env_embs[env] = per_episode_embs

        self._env_episode_state_featdicts = env_featdicts
        self._env_episode_embeddings_full = {
            env: np.stack(v, axis=0) if len(v) else np.zeros((0, 0), dtype=float)
            for env, v in env_embs.items()
        }

    def _get_inner_state_action_coverage(self, env_name: str | None = None) -> dict[str, Any]:
        """
        Compute pairwise 2D (state_i, action_j) coverage using quantile binning.

        Motivation: full (Ds+Da)-dim coverage is sparse; pairwise 2D grids are stable and comparable.

        Definitions:
          - For each (state_i, action_j) pair, define a 2D grid with `B_i * B_j` cells
            where `B_i = len(edges_i)-1` and edges are quantile edges.
          - For an episode, occupancy = number of unique visited cells across timesteps.
          - vacant_share(pair) = 1 - occupancy / (B_i*B_j)
          - episode_vacancy = mean over all pairs of vacant_share(pair)
          - env_vacancy = mean over episodes of episode_vacancy

        Args:
            env_name: If None, pre-fits bin edges for all envs and returns {"status": "ok"}.
                     If set, computes mean vacancy/coverage for that environment.

        Returns:
            For a specific env: {"mean_pairwise_vacancy": float, "mean_pairwise_coverage": float}
        """
        if env_name is None:
            for env in self.env_to_episode_paths:
                self._env_bin_edges[env] = self._fit_quantile_bin_edges_for_env(env)
            return {"status": "ok"}

        if env_name not in self._env_bin_edges:
            self._env_bin_edges[env_name] = self._fit_quantile_bin_edges_for_env(env_name)

        edges = self._env_bin_edges[env_name]
        s_edges = edges["states"]
        a_edges = edges["actions"]

        paths = self.env_to_episode_paths.get(env_name, [])
        if len(paths) == 0:
            return {"mean_pairwise_vacancy": None, "mean_pairwise_coverage": None}

        per_episode_scores = []
        for p in paths:
            df = self._read_parquet(p)
            S, A = self._get_state_action_arrays(df)

            s_bins = np.column_stack(
                [self._digitize(S[:, i], s_edges[i]) for i in range(S.shape[1])]
            )
            a_bins = np.column_stack(
                [self._digitize(A[:, j], a_edges[j]) for j in range(A.shape[1])]
            )

            vacancies = []
            for i in range(S.shape[1]):
                nb_s = max(1, len(s_edges[i]) - 1)
                for j in range(A.shape[1]):
                    nb_a = max(1, len(a_edges[j]) - 1)

                    code = s_bins[:, i].astype(np.int64) * nb_a + a_bins[:, j].astype(np.int64)
                    occupied = np.unique(code).size
                    total = nb_s * nb_a
                    vacant_share = 1.0 - (occupied / max(1, total))
                    vacancies.append(vacant_share)

            per_episode_scores.append(float(np.mean(vacancies)) if vacancies else 1.0)

        mean_vacancy = float(np.mean(per_episode_scores))
        return {
            "mean_pairwise_vacancy": mean_vacancy,
            "mean_pairwise_coverage": 1.0 - mean_vacancy,
        }

    def _get_inner_sim(self, env_name: str) -> float:
        """
        Compute within-environment episode diversity via average k-NN distance between episode embeddings.

        Returns:
            Mean Euclidean distance to the k nearest neighbors (excluding self) in standardized embedding space.
        """
        X = self._env_episode_embeddings_full.get(env_name)
        if X is None or X.shape[0] < 2:
            return 0.0

        Xs = StandardScaler().fit_transform(X)
        k = min(self.knn_k + 1, Xs.shape[0])
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(Xs)
        dists, _ = nbrs.kneighbors(Xs, return_distance=True)
        return float(dists[:, 1:].mean())

    def _get_env_importance(self, env_name: str) -> dict[str, Any]:
        """
        Estimate how "unique" a given environment is compared to the nearest other environment.

        Steps:
          1) Find nearest other env by shared state fraction:
               shared_state_frac = |S_i ∩ S_j| / |S_i|
               intersection_over_union = |S_i ∩ S_j| / |S_i ∪ S_j|
          2) Restrict embeddings to shared state variables only.
          3) Compute:
               intra = avg intra-cluster kNN distance (averaged across the two env clusters)
               inter = avg distance from points in one cluster to kNN in the other (symmetrized)
               intra_over_inter = intra / inter

        Interpretation:
          - Larger shared_state_frac => more overlap in state space (by columns).
          - Smaller intra_over_inter => clusters well-separated relative to their internal spread (more unique).

        Returns:
            {
              "nearest_env": str|None,
              "shared_state_frac": float|None,
              "intersection_over_union": float|None,
              "intra_over_inter": float|None
            }
        """
        base_states = set(self._env_state_names.get(env_name, []))
        if not base_states:
            return {
                "nearest_env": None,
                "shared_state_frac": None,
                "intersection_over_union": None,
                "intra_over_inter": None,
            }

        best_env = None
        best_shared = -1.0
        best_iou = None

        for other in self._env_state_names.keys():
            if other == env_name:
                continue
            other_states = set(self._env_state_names.get(other, []))
            inter = base_states & other_states
            shared = (len(inter) / len(base_states)) if len(base_states) > 0 else 0.0
            union = len(base_states | other_states)
            iou = (len(inter) / union) if union > 0 else 0.0

            if shared > best_shared:
                best_shared = shared
                best_iou = iou
                best_env = other

        if best_env is None:
            return {
                "nearest_env": None,
                "shared_state_frac": None,
                "intersection_over_union": None,
                "intra_over_inter": None,
            }

        shared_vars = sorted(list(base_states & set(self._env_state_names.get(best_env, []))))
        if len(shared_vars) == 0:
            return {
                "nearest_env": best_env,
                "shared_state_frac": float(best_shared),
                "intersection_over_union": float(best_iou) if best_iou is not None else None,
                "intra_over_inter": None,
            }

        X1 = self._build_embeddings_for_shared_states(env_name, shared_vars)
        X2 = self._build_embeddings_for_shared_states(best_env, shared_vars)

        if X1.shape[0] < 2 or X2.shape[0] < 2:
            return {
                "nearest_env": best_env,
                "shared_state_frac": float(best_shared),
                "intersection_over_union": float(best_iou) if best_iou is not None else None,
                "intra_over_inter": None,
            }

        X = np.vstack([X1, X2])
        Xs = StandardScaler().fit_transform(X)
        X1s = Xs[: X1.shape[0]]
        X2s = Xs[X1.shape[0] :]

        intra = 0.5 * (self._avg_intra_knn(X1s) + self._avg_intra_knn(X2s))
        inter = self._avg_inter_knn(X1s, X2s)
        ratio = float(intra / inter) if inter > 0 else None

        return {
            "nearest_env": best_env,
            "shared_state_frac": float(best_shared),
            "intersection_over_union": float(best_iou) if best_iou is not None else None,
            "intra_over_inter": ratio,
        }

    def generate_report(self) -> dict[str, Any]:
        """
        Run the full scoring pipeline and return a report.

        Returns:
            {
              "overall": dict[str, float|None],
              "per_env": pd.DataFrame  # one row per env_name
            }
        """
        self._calculate_trajectories_embeddings()
        self._get_inner_state_action_coverage()

        env_reports: list[EnvReport] = []
        for env in sorted(self.env_to_episode_paths.keys()):
            inner_knn = self._get_inner_sim(env)
            cov = self._get_inner_state_action_coverage(env)
            imp = self._get_env_importance(env)

            env_reports.append(
                EnvReport(
                    env_name=env,
                    n_episodes=len(self.env_to_episode_paths[env]),
                    env_names=self.env_group_to_env_names.get(env, []),
                    state_names=self._env_state_names.get(env, []),
                    action_names=self._env_action_names.get(env, []),
                    endogenous_names=self._env_endogenous_names.get(env, []),
                    mean_pairwise_vacancy=cov["mean_pairwise_vacancy"],
                    mean_pairwise_coverage=cov["mean_pairwise_coverage"],
                    mean_episode_embedding_knn=inner_knn,
                    nearest_env=imp["nearest_env"],
                    shared_state_frac=imp["shared_state_frac"],
                    intersection_over_union=imp["intersection_over_union"],
                    intra_over_inter=imp["intra_over_inter"],
                )
            )

        df = pd.DataFrame([r.__dict__ for r in env_reports])
        overall = {
            "n_envs": int(df.shape[0]),
            "mean_pairwise_coverage": float(df["mean_pairwise_coverage"].mean()) if len(df) else None,
            "mean_episode_embedding_knn": float(df["mean_episode_embedding_knn"].mean()) if len(df) else None,
            "mean_shared_state_frac": float(df["shared_state_frac"].dropna().mean()) if len(df) else None,
            "mean_intersection_over_union": float(df["intersection_over_union"].dropna().mean()) if len(df) else None,
            "mean_intra_over_inter": float(df["intra_over_inter"].dropna().mean()) if len(df) else None,
        }

        return {"overall": overall, "per_env": df}

    # -------- Helpers --------

    def _read_parquet(self, p: Path) -> pd.DataFrame:
        """
        Read an episode parquet file.

        Uses an in-memory cache when `cache_parquets=True` to avoid repeated disk IO.
        """
        if self.cache_parquets and p in self._parquet_cache:
            return self._parquet_cache[p]
        df = pd.read_parquet(p)
        if self.cache_parquets:
            self._parquet_cache[p] = df
        return df

    def _get_descriptions(self, df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
        """
        Extract (state_description, action_description, endogenous_description) from the episode.

        Priority:
          1) `df.iloc[0]["info"]` dict keys: state_description/action_description
          2) Episode columns: state_description/action_description
          3) Fallback: generated names s0..s(D-1), a0..a(A-1), and empty endogenous list
        """
        info0 = None
        if "info" in df.columns and len(df) > 0:
            info0 = df.iloc[0]["info"]

        if isinstance(info0, dict) and "state_description" in info0 and "action_description" in info0:
            s = list(map(str, info0["state_description"]))
            a = list(map(str, info0["action_description"]))
            e = list(map(str, info0.get("endogenous_description", [])))
            return s, a, e

        if "state_description" in df.columns and "action_description" in df.columns:
            s = list(map(str, df.iloc[0]["state_description"]))
            a = list(map(str, df.iloc[0]["action_description"]))
            e = list(map(str, df.iloc[0].get("endogenous_description", []))) if "endogenous_description" in df.columns else []
            return s, a, e

        S = np.stack(df["state"].to_list())
        A = np.stack(df["action"].to_list())
        return [f"s{i}" for i in range(S.shape[1])], [f"a{j}" for j in range(A.shape[1])], []

    def _get_state_action_arrays(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract dense numpy arrays for state/action from an episode.

        Args:
            df: Episode dataframe.

        Returns:
            (S, A) where S has shape (T, Ds) and A has shape (T, Da).
        """
        S = np.stack(df["state"].to_list()).astype(float)
        A = np.stack(df["action"].to_list()).astype(float)
        return S, A

    def _fit_quantile_bin_edges_for_env(self, env_name: str) -> dict[str, list[np.ndarray]]:
        """
        Fit quantile bin edges for each state and action dimension for a given environment.

        Edges are fit on pooled timesteps across all episodes.
        """
        all_S = []
        all_A = []
        for p in self.env_to_episode_paths[env_name]:
            df = self._read_parquet(p)
            S, A = self._get_state_action_arrays(df)
            all_S.append(S)
            all_A.append(A)

        S = np.concatenate(all_S, axis=0)
        A = np.concatenate(all_A, axis=0)

        s_edges = [self._quantile_edges(S[:, i], self.quantile_bins) for i in range(S.shape[1])]
        a_edges = [self._quantile_edges(A[:, j], self.quantile_bins) for j in range(A.shape[1])]
        return {"states": s_edges, "actions": a_edges}

    def _quantile_edges(self, x: np.ndarray, bins: int) -> np.ndarray:
        """
        Compute monotonic bin edges for 1D data using quantiles (robust for continuous variables).

        For (near-)discrete variables with few unique values, falls back to using unique values as edges.
        """
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return np.array([0.0, 1.0])

        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.array([0.0, 1.0])

        uniq = np.unique(x)
        if uniq.size <= bins:
            if uniq.size == 1:
                v = float(uniq[0])
                return np.array([v - 0.5, v + 0.5], dtype=float)
            # Use uniq as "edges" (digitize will still work; bins ~= len(uniq)-1)
            return uniq.astype(float)

        qs = np.linspace(0.0, 1.0, bins + 1)
        edges = np.quantile(x, qs)
        edges = np.unique(edges)
        if edges.size < 2:
            m = float(np.mean(x))
            return np.array([m - 0.5, m + 0.5], dtype=float)
        return edges.astype(float)

    def _digitize(self, x: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Digitize x into bin indices [0, n_bins-1] given edges returned by `_quantile_edges`."""
        if len(edges) <= 2:
            return np.zeros_like(x, dtype=np.int64)
        bins = np.digitize(x, edges[1:-1], right=True)
        return np.clip(bins, 0, len(edges) - 2).astype(np.int64)

    def _extract_ts_features(self, x: np.ndarray) -> np.ndarray:
        """
        Extract a fixed-length TS feature vector from a 1D series.

        Features include distributional stats (mean/std/quantiles), shape (skew/kurtosis),
        simple dynamics (lag-1 autocorr, linear trend slope), and energy.
        """
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.zeros(len(self.feature_names), dtype=float)

        q25, med, q75 = np.quantile(x, [0.25, 0.5, 0.75])

        ac1 = 0.0
        if x.size >= 2 and np.std(x[:-1]) > 0 and np.std(x[1:]) > 0:
            ac1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])

        slope = 0.0
        if x.size >= 3 and np.std(x) > 0:
            t = np.arange(x.size, dtype=float)
            slope = float(np.polyfit(t, x, 1)[0])

        feats = np.array(
            [
                float(np.mean(x)),
                float(np.std(x)),
                float(np.min(x)),
                float(q25),
                float(med),
                float(q75),
                float(np.max(x)),
                float(skew(x, bias=False)) if x.size >= 3 else 0.0,
                float(kurtosis(x, fisher=True, bias=False)) if x.size >= 4 else 0.0,
                float(ac1),
                float(slope),
                float(np.mean(x * x)),
            ],
            dtype=float,
        )
        return np.nan_to_num(feats)

    def _build_embeddings_for_shared_states(self, env_name: str, shared_states: list[str]) -> np.ndarray:
        """
        Build episode embeddings for a given env restricted to the provided shared state names.

        Returns:
            Array of shape (n_episodes, len(shared_states) * n_features_per_state).
        """
        featdicts = self._env_episode_state_featdicts[env_name]
        shared_states = list(shared_states)
        embs = [np.concatenate([d[s] for s in shared_states], axis=0) for d in featdicts]
        return np.stack(embs, axis=0) if embs else np.zeros((0, 0), dtype=float)

    def _avg_intra_knn(self, X: np.ndarray) -> float:
        """
        Average kNN distance within a single cluster (excluding self-neighbor).

        Args:
            X: Array of points (n, d), typically standardized.
        """
        if X.shape[0] < 2:
            return 0.0
        k = min(self.knn_k + 1, X.shape[0])
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X)
        dists, _ = nbrs.kneighbors(X)
        return float(dists[:, 1:].mean())

    def _avg_inter_knn(self, X1: np.ndarray, X2: np.ndarray) -> float:
        """
        Average cross-cluster kNN distance (symmetrized).

        Computes mean distance from each point in X1 to its k nearest neighbors in X2,
        and vice-versa, then averages the two means.
        """
        k12 = min(self.knn_k, X2.shape[0])
        nbrs2 = NearestNeighbors(n_neighbors=k12, metric="euclidean").fit(X2)
        d12, _ = nbrs2.kneighbors(X1)

        k21 = min(self.knn_k, X1.shape[0])
        nbrs1 = NearestNeighbors(n_neighbors=k21, metric="euclidean").fit(X1)
        d21, _ = nbrs1.kneighbors(X2)

        return float(0.5 * (d12.mean() + d21.mean()))

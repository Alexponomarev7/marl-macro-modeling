import json
import warnings
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

class Tokenizer:
    """
    Centralized tokenizer for state variables, actions, and environments.

    Handles token-to-ID mappings, canonicalization of state names via aliases,
    and encoding/decoding operations for use in neural network embeddings.
    """

    # State vocabulary used for variable embeddings.
    STATE_TOKENS: tuple[str, ...] = (
        "Empty",
        "Output",
        "Consumption",
        "Capital",
        "LoggedProductivity",
        "Debt",
        "InterestRate",
        "PreferenceShock",
        "CountryPremiumShock",
        "TechGrowthRate",
        "MUConsumption",
        "AR(1) Monetary Policy Shock Process",
        "AR(1) Preference Shock Process",
        "AR(1) Technology Process",
        "AR(1) Technology Shock Process",
        "Annualized Inflation Rate",
        "Annualized Interest Rate",
        "Annualized Natural Interest Rate",
        "Annualized Nominal Interest Rate",
        "Annualized Real Interest Rate",
        "Annualized Wage Inflation Rate",
        "Auxiliary Variable For Value Function",
        "Consumption Growth Rate",
        "Consumption to GDP Ratio",
        "ConsumptionPerCapita",
        "CapitalPerCapita",
        "OutputPerCapita",
        "InvestmentPerCapita",
        "ConsumptionPerEffectiveLabor",
        "CapitalPerEffectiveLabor",
        "OutputPerEffectiveLabor",
        "InvestmentPerEffectiveLabor",
        "ConsYoung",
        "ConsOld",
        "Current Account To Output Ratio",
        "Employment",
        "Exchange Rate",
        "Expected Return On Capital",
        "Expected Stochastic Discount Factor",
        "Foreign Bonds",
        "Foreign Interest Rate",
        "Foreign Price Level",
        "GovSpending",
        "LoggedGovSpending",
        "Government Spending Shock",
        "Gross Output A",
        "Gross Output B",
        "GrossReturn",
        "Growth Rate Of Money Stock",
        "HoursWorked",
        "Inflation",
        "Inflation Rate",
        "Investment",
        "Investment Growth Rate",
        "Investment to GDP Ratio",
        "Labor",
        "Lagrange Multiplier A",
        "Lagrange Multiplier B",
        "Log Output",
        "Log Tightness A",
        "Log Tightness B",
        "Log Unemployment",
        "Log Vacancies",
        "Log Wages",
        "Marginal Costs",
        "MarginalProductCapital",
        "Market Tightness",
        "Markup",
        "Matches",
        "Meeting Rate Between Firms And Workers",
        "Money Growth",
        "Money Growth Annualized",
        "Money Stock",
        "MoneySupply",
        "Natural Interest Rate",
        "Natural Output",
        "Natural Real Wage",
        "Net Exports",
        "Nominal Interest Rate",
        "Nominal Money Stock",
        "Nominal Wage",
        "Output Deviation From Steady State",
        "Output Gap",
        "Output Growth",
        "Output Growth Rate",
        "Output Minus Consumption",
        "Price Inflation",
        "Price Level",
        "Productivity",
        "Real Consumption",
        "Real Interest Rate",
        "Real Money Stock",
        "Real Output",
        "Real Return On Capital",
        "Real Wage",
        "Real Wage Gap",
        "Wage",
        "WagePerEffectiveLabor",
        "Savings",
        "Risk-Free Rate",
        "TaxRate",
        "Technology Shock",
        "Trade Balance to Output Ratio",
        "TradeBalance",
        "Unemployment Rate",
        "Utility",
        "Vacancies",
        "Value Function",
        "Volatility",
        "Wage Inflation",
        "LoggedVolatility",
        "MeetingRate",
        "MonetaryShock",
    )

    # State aliases for canonicalization across environments
    STATE_ALIASES: dict[str, str] = {
        # High-confidence spelling variants (same meaning, different formatting)
        "Capital Stock": "Capital",
        "Country Premium Shock": "CountryPremiumShock",
        "Hours Worked": "HoursWorked",
        "Interest Rate": "InterestRate",
        "Log TFP": "LoggedProductivity",
        "Marginal Utility": "MUConsumption",
        "Marginal Utility of Consumption": "MUConsumption",
        "Preference Shock": "PreferenceShock",
        "Technology": "LoggedProductivity",
        "Technology Growth Rate": "TechGrowthRate",
        "Trade Balance": "TradeBalance",
        "Trade Balance To Output Ratio": "Trade Balance to Output Ratio",
        "Government Spending": "GovSpending",
        # Some models use long_name='Total Factor Productivity' for a state that is
        # actually logged productivity in our pipeline/configs.
        "Total Factor Productivity": "LoggedProductivity",
        # Some `.mod` files use lower-case long_name values even when the TeX/header is title-cased
        # (e.g. `Ramsey.mod`: long_name='consumption'). Canonicalize those back to our column names.
        "consumption": "Consumption",
        "capital": "Capital",
        "output": "Output",
        "investment": "Investment",
        # Per-capita variables (from Dynare long_name format)
        "consumption per capita": "ConsumptionPerCapita",
        "capital per capita": "CapitalPerCapita",
        "output per capita": "OutputPerCapita",
        "investment per capita": "InvestmentPerCapita",
        # Per-effective-labor variables
        "consumption per effective labor": "ConsumptionPerEffectiveLabor",
        "capital per effective labor": "CapitalPerEffectiveLabor",
        "output per effective labor": "OutputPerEffectiveLabor",
        "investment per effective labor": "InvestmentPerEffectiveLabor",
        "wage per effective labor unit": "WagePerEffectiveLabor",
        "wage per effective labor": "WagePerEffectiveLabor",
        # OLG model variables
        "consumption young": "ConsYoung",
        "consumption old": "ConsOld",
        "savings per effective worker": "Savings",
        "capital stock (end of period)": "Capital",
        "output per effective worker": "Output",
        "wage rate": "Wage",
        "rental rate of capital": "InterestRate",
        # Other common lowercase long_name variants
        "labor/population": "Labor",
        "technology level": "Productivity",
        "real interest rate": "InterestRate",
        "gross return on capital": "GrossReturn",
        "marginal product of capital": "MarginalProductCapital",
        "logged tfp": "LoggedProductivity",
        "tfp level": "Productivity",
        "aggregate consumption": "Consumption",
        "aggregate capital": "Capital",
        "aggregate output": "Output",
        "Logged TFP (transitory)": "LoggedProductivity",
        "Log Volatility": "LoggedVolatility",
        "meeting rate firms and workers": "MeetingRate",
    }

    ACTION_TOKENS: tuple[str, ...] = (
        "Empty",
        "Investment",
        "Savings",
        "savings per effective worker",
        "Consumption",
        "ConsumptionPerEffectiveLabor",
        "ConsumptionPerCapita",
        "consumption_fraction",
        "consumption_rate",
        "HoursWorked",
        "Consumers",  # from marl_rbc_with_irrational_behavior.py
        "Firms",  # from marl_rbc_with_irrational_behavior.py
        "Government",  # from marl_rbc_with_irrational_behavior.py
        "Labor",  # from marl_rbc_with_irrational_behavior.py
        "Nominal Interest Rate",
        "GovSpending",  # from RBC_state_dependent_GIRF
        "Growth Rate Of Money Stock",  # from McCandless_2008_Chapter_13
        "gov_spending_change",
        "investment_rate",
        "leisure",
        "money_supply_change",
        "tax_rate_change",
    )

    ACTION_ALIASES: dict[str, str] = {
        "Real Consumption": "Consumption",
        "Capital": "Savings",  # In OLG models: (1 + n) * (1 + g) * Capital = Savings
    }

    ENV_MAPPING: dict[str, int] = {
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
        "Ramsey_base": 17,
        "Ramsey_crra": 18,
        "Ramsey_cara": 19,
        "Ramsey_upgrade": 20,
        "OLG": 21,
        "RBC_baseline_pf": 22,
        "RBC_baseline_stoch": 23,
    }

    def __init__(self):
        """Initialize tokenizer with validated mappings."""
        self._state_mapping = {name: i for i, name in enumerate(self.STATE_TOKENS)}
        self._action_mapping = {name: i for i, name in enumerate(self.ACTION_TOKENS)}

        # Create normalized lookup dictionaries for case-insensitive, space-ignoring matching
        self._normalized_state_mapping = {
            self._normalize_key(key): value
            for key, value in self._state_mapping.items()
        }
        self._normalized_action_mapping = {
            self._normalize_key(key): value
            for key, value in self._action_mapping.items()
        }
        self._normalized_state_aliases = {
            self._normalize_key(key): value
            for key, value in self.STATE_ALIASES.items()
        }
        self._normalized_action_aliases = {
            self._normalize_key(key): value
            for key, value in self.ACTION_ALIASES.items()
        }

        self._validate_token_mapping(self._state_mapping, mapping_name="STATE_MAPPING")
        self._validate_token_mapping(self._action_mapping, mapping_name="ACTION_MAPPING")

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Normalize a key for matching: lowercase and remove spaces."""
        return key.lower().replace(" ", "")

    @staticmethod
    def _validate_token_mapping(mapping: dict[str, int], *, mapping_name: str) -> None:
        """Ensure token->id mapping is safe for nn.Embedding and stable to use as IDs."""
        if "Empty" not in mapping or mapping["Empty"] != 0:
            raise ValueError(f"{mapping_name} must include 'Empty': 0")
        ids = list(mapping.values())
        if len(ids) != len(set(ids)):
            raise ValueError(f"{mapping_name} has duplicate IDs: {mapping}")
        if min(ids) != 0 or max(ids) != len(mapping) - 1:
            raise ValueError(
                f"{mapping_name} IDs must be contiguous 0..{len(mapping)-1}, got {sorted(ids)}"
            )

    def canonical_state_name(self, name: str) -> str:
        """Canonicalize a state variable name across environments (case-insensitive, ignores spaces)."""
        # First try exact match in state tokens
        if name in self._state_mapping:
            return name

        # Try exact match in aliases
        if name in self.STATE_ALIASES:
            return self.STATE_ALIASES[name]

        # Try normalized match in state tokens
        normalized_name = self._normalize_key(name)
        for orig_token in self._state_mapping.keys():
            if self._normalize_key(orig_token) == normalized_name:
                return orig_token

        # Try normalized match in aliases
        for orig_key, orig_value in self.STATE_ALIASES.items():
            if self._normalize_key(orig_key) == normalized_name:
                return orig_value

        # If no match found, return the name as-is
        return name

    def canonical_action_name(self, name: str) -> str:
        """Canonicalize an action variable name across environments (case-insensitive, ignores spaces)."""
        # First try exact match in action tokens
        if name in self._action_mapping:
            return name

        # Try exact match in aliases
        if name in self.ACTION_ALIASES:
            return self.ACTION_ALIASES[name]

        # Try normalized match in action tokens
        normalized_name = self._normalize_key(name)
        for orig_token in self._action_mapping.keys():
            if self._normalize_key(orig_token) == normalized_name:
                return orig_token

        # Try normalized match in aliases
        for orig_key, orig_value in self.ACTION_ALIASES.items():
            if self._normalize_key(orig_key) == normalized_name:
                return orig_value

        # If no match found, return the name as-is
        return name

    def state_token_id(self, name: str) -> int:
        """Map a state variable name (possibly an alias) to a stable token id (case-insensitive, ignores spaces)."""
        canon = self.canonical_state_name(name)

        # Try exact match first
        if canon in self._state_mapping:
            return self._state_mapping[canon]

        # Try normalized match
        normalized_canon = self._normalize_key(canon)
        if normalized_canon in self._normalized_state_mapping:
            return self._normalized_state_mapping[normalized_canon]

        # Try finding by normalized match in original tokens
        for orig_token, token_id in self._state_mapping.items():
            if self._normalize_key(orig_token) == normalized_canon:
                return token_id

        raise KeyError(
            f"Unknown state token '{name}' (canonical '{canon}'). "
            f"Known tokens: {list(self._state_mapping.keys())}"
        )

    def action_token_id(self, name: str) -> int:
        """
        Map an action variable name (possibly an alias) to a stable token id (case-insensitive, ignores spaces).
        """
        canon = self.canonical_action_name(name)

        # Try exact match first
        if canon in self._action_mapping:
            return self._action_mapping[canon]

        # Try normalized match
        normalized_canon = self._normalize_key(canon)
        if normalized_canon in self._normalized_action_mapping:
            return self._normalized_action_mapping[normalized_canon]

        # Try finding by normalized match in original tokens
        for orig_token, token_id in self._action_mapping.items():
            if self._normalize_key(orig_token) == normalized_canon:
                return token_id

        raise KeyError(
            f"Unknown action token '{name}' (canonical '{canon}'). "
            f"Known tokens: {list(self._action_mapping.keys())}"
        )

    def decode_env_name(self, env_name: str) -> int:
        """Decode an environment name to its ID."""
        prefix = env_name.rsplit('_', 1)[0]
        if prefix.endswith('_config'):
            prefix = prefix.removesuffix('_config')
        if prefix not in self.ENV_MAPPING:
            return 0  # Default to 0 if not found
        return self.ENV_MAPPING[prefix]

    def state_encoder(self, x):
        """Encode state values (placeholder for future implementation)."""
        return x

    def action_encoder(self, x):
        """Encode action values (placeholder for future implementation)."""
        return x

    @property
    def state_mapping(self) -> dict[str, int]:
        """Get the state token-to-ID mapping."""
        return self._state_mapping.copy()

    @property
    def action_mapping(self) -> dict[str, int]:
        """Get the action token-to-ID mapping."""
        return self._action_mapping.copy()

    @property
    def num_state_tokens(self) -> int:
        """Get the number of state tokens."""
        return len(self._state_mapping)

    @property
    def num_action_tokens(self) -> int:
        """Get the number of action tokens."""
        return len(self._action_mapping)

    @property
    def num_tasks(self) -> int:
        """Get the number of tasks (environments)."""
        # num_tasks = max task ID + 1 (since IDs are 0-indexed)
        return max(self.ENV_MAPPING.values()) + 1 if self.ENV_MAPPING else 1


# Create a default tokenizer instance for use across the module
_default_tokenizer = Tokenizer()

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
        self.tokenizer = _default_tokenizer
        self.task_ids = [self.tokenizer.decode_env_name(item['env_name']) for item in self.metadata]

        # todo: Encoders of environment state and action
        self.state_encoder = self.tokenizer.state_encoder
        self.action_encoder = self.tokenizer.action_encoder

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
        model_params_values = model_params_values[:self.max_model_params_dim]

        # Pad states to max_state_dim
        states = self.pad_dim(states, self.max_state_dim)
        state_description = data.iloc[0]["info"]["state_description"]
        action_description = data.iloc[0]["info"]["action_description"]
        endogenous_description = data.iloc[0]["info"]["endogenous_description"]
        # Truncate descriptions if they exceed max dimensions, then pad to max dimensions
        state_description = state_description[:self.max_state_dim]
        action_description = action_description[:self.max_action_dim]
        endogenous_description = endogenous_description[:self.max_endogenous_dim]
        states_info = torch.tensor([self.tokenizer.state_token_id(state) for state in state_description] + [0] * (self.max_state_dim - len(state_description)), dtype=torch.long)
        actions_info = torch.tensor([self.tokenizer.action_token_id(action) for action in action_description] + [0] * (self.max_action_dim - len(action_description)), dtype=torch.long)
        endogenous_info = torch.tensor([self.tokenizer.state_token_id(endogenous) for endogenous in endogenous_description] + [0] * (self.max_endogenous_dim - len(endogenous_description)), dtype=torch.long)
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
    intra: float | None  # avg intra-cluster kNN distance
    inter: float | None  # avg inter-cluster kNN distance


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
            # Resolve relative paths relative to dataset_path
            if not p.is_absolute():
                # Paths in metadata may be relative to project root (e.g., "data/processed/train/file.parquet")
                # If the path contains the dataset_path name, extract just the filename
                p_str = str(p)
                if self.dataset_path.name in p_str:
                    # Extract just the filename (last component after the dataset_path name)
                    # This handles cases like "data/processed/train/file.parquet" -> "file.parquet"
                    parts = p_str.split(self.dataset_path.name)
                    if len(parts) > 1:
                        # Take everything after dataset_path name and remove leading slashes
                        filename_part = parts[-1].lstrip("/")
                        p = Path(filename_part) if filename_part else Path(p.name)
                # Resolve relative to dataset_path
                p = (self.dataset_path / p).resolve()
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
              "intra_over_inter": float|None,
              "intra": float|None,
              "inter": float|None
            }
        """
        base_states = set(self._env_state_names.get(env_name, []))
        if not base_states:
            return {
                "nearest_env": None,
                "shared_state_frac": None,
                "intersection_over_union": None,
                "intra_over_inter": None,
                "intra": None,
                "inter": None,
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
                "intra": None,
                "inter": None,
            }

        shared_vars = sorted(list(base_states & set(self._env_state_names.get(best_env, []))))
        if len(shared_vars) == 0:
            return {
                "nearest_env": best_env,
                "shared_state_frac": float(best_shared),
                "intersection_over_union": float(best_iou) if best_iou is not None else None,
                "intra_over_inter": None,
                "intra": None,
                "inter": None,
            }

        X1 = self._build_embeddings_for_shared_states(env_name, shared_vars)
        X2 = self._build_embeddings_for_shared_states(best_env, shared_vars)

        if X1.shape[0] < 2 or X2.shape[0] < 2:
            return {
                "nearest_env": best_env,
                "shared_state_frac": float(best_shared),
                "intersection_over_union": float(best_iou) if best_iou is not None else None,
                "intra_over_inter": None,
                "intra": None,
                "inter": None,
            }

        X = np.vstack([X1, X2])
        Xs = StandardScaler().fit_transform(X)
        X1s = Xs[: X1.shape[0]]
        X2s = Xs[X1.shape[0] :]

        # Check if embeddings have any variance (all points identical would cause issues)
        eps = 1e-10
        if Xs.shape[0] > 1:
            # Check if all points are identical (within numerical precision)
            point_std = np.std(Xs, axis=0)
            if np.all(point_std < eps):
                # All embeddings are identical - cannot compute meaningful ratio
                return {
                    "nearest_env": best_env,
                    "shared_state_frac": float(best_shared),
                    "intersection_over_union": float(best_iou) if best_iou is not None else None,
                    "intra_over_inter": None,
                    "intra": None,
                    "inter": None,
                }

        intra = 0.5 * (self._avg_intra_knn(X1s) + self._avg_intra_knn(X2s))
        inter = self._avg_inter_knn(X1s, X2s)

        # Add small epsilon to avoid division by zero and handle numerical precision issues
        # If inter is exactly 0 or too small, it means embeddings are too similar
        # If intra is 0, it means episodes within a model are identical
        if inter <= eps:
            # If inter is too small, models are too similar - return None to indicate invalid comparison
            ratio = None
        else:
            ratio = float(intra / inter)

        return {
            "nearest_env": best_env,
            "shared_state_frac": float(best_shared),
            "intersection_over_union": float(best_iou) if best_iou is not None else None,
            "intra_over_inter": ratio,
            "intra": float(intra),
            "inter": float(inter),
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
                    intra=imp["intra"],
                    inter=imp["inter"],
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

        # Check variance before computing skew/kurtosis to avoid precision warnings
        # If data has no variance (nearly identical values), these statistics are not meaningful
        x_std = np.std(x)
        skew_val = 0.0
        if x.size >= 3 and x_std > 1e-10:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Precision loss.*")
                skew_val = float(skew(x, bias=False))

        kurt_val = 0.0
        if x.size >= 4 and x_std > 1e-10:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Precision loss.*")
                kurt_val = float(kurtosis(x, fisher=True, bias=False))

        feats = np.array(
            [
                float(np.mean(x)),
                float(np.std(x)),
                float(np.min(x)),
                float(q25),
                float(med),
                float(q75),
                float(np.max(x)),
                skew_val,
                kurt_val,
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

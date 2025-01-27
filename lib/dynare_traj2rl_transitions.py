import importlib
import shutil
import tempfile
from typing import Callable, Optional, cast
from multiprocessing import Pool, cpu_count
from collections import deque
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
import pickle
import os
import re
from pathlib import Path

import yaml
import subprocess

from loguru import logger
import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import traceback

from research.utils import PathStorage
from lib.dataset import Tokenizer

# Use the shared state aliases for column renaming
_tokenizer = Tokenizer()
_COLUMN_ALIASES = Tokenizer.STATE_ALIASES


class StateAccessor:
    def __init__(self, state_columns: list[str], buffer_size: int = 10):
        self.state_columns = state_columns
        self.buffer = deque(maxlen=buffer_size)
        self.raw_columns = [column[0] if isinstance(column, list) else column for column in state_columns]

    def get_columns(self) -> list[str]:
        return self.raw_columns

    def __call__(self, row: pd.Series) -> np.ndarray:
        state = []
        for column in self.state_columns:
            if isinstance(column, list):
                state_column, shift = column
                assert int(shift) < 0, "shift should be negative"
                state.append(float(self.buffer[int(shift)][state_column]))
            else:
                state.append(float(row[column]))

        self.buffer.append(pd.Series(row))
        return np.array(state)


def sample_from_range(range_values: list[float]) -> float:
    return np.random.random() * (range_values[1] - range_values[0]) + range_values[0]


def _generate_shock_params(
    shock_settings: dict,
    periods: int,
    prefix: str,
    shock_count_name: str,
    max_shocks: int = 6
) -> dict:
    """
    Generate shock parameters for a single shock type.

    Args:
        shock_settings: Settings for this shock type (num_shocks, period_range, value_range).
        periods: Total simulation periods.
        prefix: Prefix for parameter names (e.g., 'productivity_shock').
        shock_count_name: Name for the shock count parameter.
        max_shocks: Maximum number of shocks to generate.

    Returns:
        Dictionary with shock parameters.
    """
    params = {}

    num_shocks = shock_settings.get("num_shocks", 0)
    params[shock_count_name] = num_shocks

    period_range = shock_settings.get("period_range", [1, periods])
    period_start = max(1, period_range[0])
    period_end = min(periods, period_range[1])
    periods_available = list(range(period_start, period_end + 1))

    actual_num_shocks = min(num_shocks, len(periods_available))
    if actual_num_shocks > 0:
        shock_periods = sorted(np.random.choice(
            periods_available,
            size=actual_num_shocks,
            replace=False
        ))
    else:
        shock_periods = []

    value_range = shock_settings.get("value_range", [-0.05, 0.05])

    for i in range(max_shocks):
        if i < len(shock_periods):
            period = int(shock_periods[i])
            value = sample_from_range(value_range)
        else:
            period = 1
            value = 0.0

        params[f"{prefix}_period_{i+1}"] = period
        params[f"{prefix}_value_{i+1}"] = value

    return params


def _generate_all_shocks(
    shocks_config: dict,
    periods: int,
    max_shocks_per_type: int = 6
) -> dict:
    """
    Generate parameters for all shock types.

    Args:
        shocks_config: Dictionary where keys are shock names and values are settings.
        periods: Total simulation periods.
        max_shocks_per_type: Maximum number of shocks per type.

    Returns:
        Dictionary with all shock parameters for Dynare.
    """
    all_params = {}

    for shock_name, shock_settings in shocks_config.items():
        shock_params = _generate_shock_params(
            shock_settings=shock_settings,
            periods=periods,
            prefix=f"{shock_name}_shock",
            shock_count_name=f"num_{shock_name}_shocks",
            max_shocks=max_shocks_per_type
        )
        all_params.update(shock_params)

    return all_params


def generate_parameter_combinations(model_settings: dict, num_samples: int, model_name: str = None) -> tuple[list[list[str]], list[dict]]:
    parameter_combinations = []
    parameter_values = []

    for _ in range(num_samples):
        current_combination = []
        current_values = {}

        # Handle periods separately since it's not a range
        if "periods" in model_settings:
            current_combination.append(f"-Dperiods={model_settings['periods']}")
            current_values["periods"] = model_settings["periods"]

        # Sample from parameter ranges
        if "parameter_ranges" in model_settings:
            for param, range_values in model_settings["parameter_ranges"].items():
                value = sample_from_range(range_values)
                current_combination.append(f"-D{param}={value}")
                current_values[param] = value

        # Handle direct parameters (not in parameter_ranges) - these are flags or other settings
        for param, value in model_settings.items():
            if param not in ["periods", "num_samples", "column_names", "shocks", "parameter_ranges"]:
                if isinstance(value, list) and len(value) == 2:
                    # It's a range, sample from it
                    sampled_value = sample_from_range(value)
                    # For boolean flags [0, 1], round to 0 or 1 and ensure it's an integer
                    if sorted(value) == [0, 1]:
                        sampled_value = int(1 if sampled_value >= 0.5 else 0)
                    current_combination.append(f"-D{param}={sampled_value}")
                    current_values[param] = sampled_value
                else:
                    # It's a direct value - ensure integers are passed as integers
                    if isinstance(value, (int, float)) and value in [0, 1]:
                        value = int(value)
                    current_combination.append(f"-D{param}={value}")
                    current_values[param] = value

        if "shocks" in model_settings:
            shock_params = _generate_all_shocks(
                shocks_config=model_settings["shocks"],
                periods=model_settings.get("periods", 100)
            )
            for key, value in shock_params.items():
                current_combination.append(f"-D{key}={value}")
                current_values[key] = value

        parameter_combinations.append(current_combination)
        parameter_values.append(current_values)

    return parameter_combinations, parameter_values


def get_reward_object(reward_object_path: str) -> Optional[Callable]:
    """Imports and returns a reward object from a specified path.

    This function dynamically imports a reward object (typically a class or function)
    from a given module path. The path should be in the format 'module_name.class_name'.

    Args:
        reward_object_path (str): The dotted path to the reward object,
            e.g., 'module_name.class_name'.

    Returns:
        Callable: The imported reward object (class or function). Returns None if
            the import fails due to an ImportError or AttributeError.

    """
    reward_object = None
    try:
        module_name, class_name = reward_object_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        reward_object = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Error importing reward object: {e}")

    assert reward_object is not None, f"reward object is not imported {reward_object_path=}"
    return reward_object


def run_model(
    input_file: Path,
    output_file: Path,
    output_params_file: Path,
    parameters: list[str],
    max_retries: int = 3,
) -> None:
    """Run a Dynare model with specified parameters and save results.

    Args:
        input_file: Path to the Dynare .mod file
        output_file: Path to save the output CSV
        periods: Number of simulation periods
        parameters: List of parameter strings to pass to Dynare
        max_retries: Maximum number of retry attempts
    """
    retries = 0
    while retries < max_retries:
        print(f"Running model: {input_file} (attempt {retries + 1})")
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                input_tmp_file = Path(tmp_dir) / input_file.name
                shutil.copy(input_file, input_tmp_file)

                # Run Dynare model
                cmd: list[str] = [
                    "octave",
                    "--eval",
                    f"""
                    addpath {os.environ["DYNARE_PATH"]};
                    cd {input_tmp_file.parent};
                    dynare {input_tmp_file.name} {' '.join(parameters)};
                    oo_simul = oo_.endo_simul';
                    var_names = M_.endo_names_tex;
                    param_names = M_.param_names;
                    param_values = M_.params;

                    fid = fopen('{output_file}', 'w');
                    fprintf(fid, '%s,', var_names{{1:end-1}});
                    fprintf(fid, '%s\\n', var_names{{end}});
                    fclose(fid);

                    dlmwrite('{output_file}', oo_simul, '-append');
                    fid = fopen('{output_params_file}', 'w');
                    for i = 1:length(param_names)
                        fprintf(fid, '%s: %f\\n', char(param_names(i)), param_values(i));
                    end
                    fclose(fid);
                    """
                ]

                process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            logger.info(f"Running command: {subprocess.list2cmdline(cmd)}")
            print(f"Command: {subprocess.list2cmdline(cmd)}")
            print(f"Parameters: {' '.join(parameters)}")
            print(f"Return code: {process.returncode}")

            if process.returncode != 0:
                stdout_preview = process.stdout[-2000:] if len(process.stdout) > 2000 else process.stdout
                stderr_preview = process.stderr[-2000:] if len(process.stderr) > 2000 else process.stderr
                error_msg = f"Dynare failed with return code {process.returncode}:\n\nSTDOUT (last 2000 chars):\n{stdout_preview}\n\nSTDERR (last 2000 chars):\n{stderr_preview}\n\nFull STDOUT length: {len(process.stdout)}, Full STDERR length: {len(process.stderr)}"
                print("="*80)
                print("DYNARE ERROR OUTPUT:")
                print(error_msg)
                print("="*80)
                import sys
                sys.stdout.flush()
                sys.stderr.flush()
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            print(f"Model {input_file} completed successfully.")
            return

        except Exception as e:
            print(f"\n{'='*80}")
            print(f"Error running model {input_file} (attempt {retries + 1}/{max_retries}):")
            print(f"Error message: {str(e)}")
            print(f"Parameters used: {' '.join(parameters)}")
            print("Stack trace:")
            traceback.print_exc()
            print(f"{'='*80}\n")

            retries += 1
            if retries < max_retries:
                print(f"Retrying model {input_file}...")
            else:
                print(f"Failed to run model {input_file} after {max_retries} attempts.")
                print(f"Final parameters that failed: {' '.join(parameters)}")

    if retries == max_retries:
        raise RuntimeError(f"Failed to run model {input_file} after {max_retries} attempts.")


def process_model_combination(args):
    _, input_file, base_name, combination, values, raw_data_dir = args
    output_file = os.path.join(os.getcwd(), raw_data_dir, base_name + "_raw.csv")
    output_params_file = os.path.join(os.getcwd(), raw_data_dir, base_name + "_params.yaml")
    config_file = os.path.join(os.getcwd(), raw_data_dir, base_name + "_config.yml")

    run_model(Path(input_file), Path(output_file), Path(output_params_file), combination)
    print(f"Output saved to {output_file}")
    print(f"Config saved to {config_file}")
    print(f"Params saved to {output_params_file}")


def run_models(config: dict, raw_data_dir: Path) -> list[tuple[Path, Path]]:
    output_files = []
    tasks = []

    for model_name, model_config in config.items():
        model_settings = model_config["dynare_model_settings"]
        num_samples = model_settings["num_samples"]
        parameter_combinations, parameter_values = generate_parameter_combinations(model_settings, num_samples)

        for i, (combination, values) in enumerate(zip(parameter_combinations, parameter_values)):
            input_file = os.path.join(PathStorage().dynare_configs_root, model_name + ".mod")
            base_name = "_".join([model_name, f"config_{i}"])
            output_file = os.path.join(os.getcwd(), raw_data_dir, base_name + "_raw.csv")
            output_params_file = os.path.join(os.getcwd(), raw_data_dir, base_name + "_params.yaml")
            output_files.append((Path(output_file), Path(output_params_file)))
            task = (model_name, input_file, base_name, combination, values, raw_data_dir)
            tasks.append(task)

    num_processes = min(min(cpu_count(), len(tasks)), 32)
    print(f"Running {len(tasks)} tasks using {num_processes} processes")

    with Pool(processes=num_processes) as pool:
        pool.map(process_model_combination, tasks)

    return output_files


def dynare_trajectories2rl_transitions(
    input_data_path: Path,
    state_accessor: StateAccessor,
    endogenous_accessor: StateAccessor,
    action_columns: list[str],
    reward_fn: Callable,
    reward_kwargs: dict,
    discount_factor: float,
    model_params: dict,
    column_renames: dict[str, str] | None = None,
    mod_file_path: Path | None = None,
) -> pd.DataFrame:
    """Converts Dynare trajectories into reinforcement learning transitions.

    Args:
        input_data_path (str): Path to the input CSV file containing Dynare data.
        state_columns (list[str]): List of column names representing the state variables.
        action_columns (list[str]): List of column names representing the action variables.
        reward_func (Callable): A function that computes the reward.
        reward_kwargs (dict): Additional keyword arguments for the reward function.
        mod_file_path: Optional path to .mod file for parsing TeX headers.

    Returns:
        pd.DataFrame: A DataFrame containing the transitions.
    """
    data = pd.read_csv(input_data_path)
    original_columns = set(data.columns)

    if column_renames:
        # Only rename columns that actually exist in the file (safe no-op otherwise).
        data = data.rename(columns={k: v for k, v in column_renames.items() if k in data.columns})

    data_cols = set(data.columns)

    # Build TeX-to-canonical mapping from .mod file if available
    tex_to_canonical: dict[str, str] = {}
    canonical_to_tex: dict[str, list[str]] = {}  # canonical -> list of TeX headers
    long_name_to_canonical: dict[str, str] = {}
    if mod_file_path and mod_file_path.exists():
        def _parse_mod_symbol_tex_to_long(mod_text: str) -> dict[str, str]:
            """Parse .mod file to map TeX headers to long names.

            Only parses `var` and `varexo` declarations, not `parameters`, since CSV files
            only contain variable data.
            """
            mapping: dict[str, str] = {}
            # Extract only var and varexo sections (CSV files don't contain parameters)
            var_section_pattern = re.compile(
                r"\b(?:var|varexo)\b.*?(?=\b(?:varexo|parameters|model|steady_state_model)\b|$)",
                re.DOTALL
            )
            var_sections = var_section_pattern.findall(mod_text)
            var_text = "\n".join(var_sections)

            pat = re.compile(
                r"\b(?P<sym>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<tex>\$[^$]+\$)\s*\(long_name='(?P<long>[^']+)'\)"
            )
            for m in pat.finditer(var_text):
                sym = m.group("sym")
                tex = m.group("tex")
                long_name = m.group("long")
                # Symbol-based mappings always take precedence (add first)
                mapping[sym] = long_name
                mapping[sym.upper()] = long_name  # Add uppercase variant
                mapping[sym.lower()] = long_name  # Add lowercase variant
                mapping["{" + sym + "}"] = long_name
                mapping["{" + sym.upper() + "}"] = long_name  # Add uppercase variant
                mapping["{" + sym.lower() + "}"] = long_name  # Add lowercase variant
                # TeX mappings only if not already set (preserves symbol-based mappings)
                if tex not in mapping:
                    mapping[tex] = long_name
                tex_stripped = tex.strip("$")
                if tex_stripped not in mapping:
                    mapping[tex_stripped] = long_name
                if tex_stripped.upper() not in mapping:
                    mapping[tex_stripped.upper()] = long_name
                if tex_stripped.lower() not in mapping:
                    mapping[tex_stripped.lower()] = long_name
            return mapping

        try:
            mod_text = mod_file_path.read_text(errors="ignore")
            sym_tex_to_long = _parse_mod_symbol_tex_to_long(mod_text)
            # Map long_name -> canonical, applying full alias chain
            def resolve_canonical(name: str) -> str:
                """Resolve to canonical name, applying full alias chain."""
                seen = set()
                current = name
                while current in _COLUMN_ALIASES and current not in seen:
                    seen.add(current)
                    current = _COLUMN_ALIASES[current]
                return current

            # Build mapping from TeX headers to canonical names
            for tex_key, long_name in sym_tex_to_long.items():
                canonical = resolve_canonical(long_name)
                tex_to_canonical[tex_key] = canonical
                long_name_to_canonical[long_name] = canonical
                # Store all TeX headers for this canonical
                if canonical not in canonical_to_tex:
                    canonical_to_tex[canonical] = []
                if tex_key not in canonical_to_tex[canonical]:
                    canonical_to_tex[canonical].append(tex_key)
                # Also map symbol name directly
                if tex_key.startswith("{") and tex_key.endswith("}"):
                    sym_name = tex_key[1:-1]
                    if sym_name and sym_name.isalnum():
                        tex_to_canonical[sym_name] = canonical
                        if sym_name not in canonical_to_tex[canonical]:
                            canonical_to_tex[canonical].append(sym_name)
        except Exception:
            pass  # If parsing fails, just proceed without TeX mapping

    def _resolve_column_name(requested: str) -> str:
        """
        Resolve a requested column name against the dataframe columns.

        Priority:
          1) Exact match (requested exists)
          2) Canonical match via `canonical_state_name(requested)`
          3) Reverse-alias match: some alias that maps to `requested`
          4) TeX header match: if mod file was parsed, check if requested canonical maps to a TeX header in data
        """
        if requested in data_cols:
            return requested

        canon = _tokenizer.canonical_state_name(requested)
        if canon in data_cols:
            return canon

        # Reverse-alias lookup (canonical -> an existing alias in the data)
        reverse_candidates = [k for k, v in _COLUMN_ALIASES.items() if v == requested and k in data_cols]
        if len(reverse_candidates) == 1:
            return reverse_candidates[0]
        if len(reverse_candidates) > 1:
            raise KeyError(
                f"Ambiguous column alias for '{requested}'. "
                f"Candidates present in file: {sorted(reverse_candidates)}"
            )

        # TeX header lookup: if we have a mod file mapping, check if requested canonical
        # corresponds to a TeX header that exists in the data
        if tex_to_canonical:
            # Find TeX headers that map to the canonical name of requested
            if canonical_to_tex and canon in canonical_to_tex:
                tex_candidates = [tex for tex in canonical_to_tex[canon] if tex in data_cols]
            else:
                tex_candidates = [tex for tex, can in tex_to_canonical.items() if can == canon and tex in data_cols]
            if len(tex_candidates) == 1:
                return tex_candidates[0]
            if len(tex_candidates) > 1:
                raise KeyError(
                    f"Ambiguous TeX header for '{requested}' (canonical: '{canon}'). "
                    f"Candidates present in file: {sorted(tex_candidates)}"
                )
            # Also try the original requested name (before canonicalization)
            if requested != canon:
                # First check if requested is a long_name that maps to a canonical
                if requested in long_name_to_canonical:
                    requested_canon = long_name_to_canonical[requested]
                    if canonical_to_tex and requested_canon in canonical_to_tex:
                        tex_candidates = [tex for tex in canonical_to_tex[requested_canon] if tex in data_cols]
                    else:
                        tex_candidates = [tex for tex, can in tex_to_canonical.items() if can == requested_canon and tex in data_cols]
                    if len(tex_candidates) == 1:
                        return tex_candidates[0]
                    if len(tex_candidates) > 1:
                        raise KeyError(
                            f"Ambiguous TeX header for '{requested}' (canonical: '{requested_canon}'). "
                            f"Candidates present in file: {sorted(tex_candidates)}"
                        )
                # Also check if requested directly maps to a TeX header
                tex_candidates = [tex for tex, can in tex_to_canonical.items() if can == requested and tex in data_cols]
                if len(tex_candidates) == 1:
                    return tex_candidates[0]
                if len(tex_candidates) > 1:
                    raise KeyError(
                        f"Ambiguous TeX header for '{requested}'. "
                        f"Candidates present in file: {sorted(tex_candidates)}"
                    )

        return requested

    def _resolve_accessor_columns(accessor: StateAccessor) -> None:
        """Resolve accessor columns in-place to match `data` columns."""
        resolved: list[str | list] = []
        for col in accessor.state_columns:
            if isinstance(col, list):
                state_col, shift = col
                resolved.append([_resolve_column_name(str(state_col)), shift])
            else:
                resolved.append(_resolve_column_name(str(col)))
        accessor.state_columns = resolved  # type: ignore[assignment]
        accessor.raw_columns = [c[0] if isinstance(c, list) else c for c in resolved]

    # Resolve state/action/endogenous columns to avoid KeyError after renaming.
    _resolve_accessor_columns(state_accessor)
    _resolve_accessor_columns(endogenous_accessor)
    action_columns = [_resolve_column_name(c) for c in action_columns]

    # Resolve reward kwargs that refer to dataframe columns (reward fns use these to index `data`)
    reward_kwargs = dict(reward_kwargs)
    for key in (
        "target_column",
        "consumption_column",
        "labor_column",
        "consumption_young_column",
        "consumption_old_column",
    ):
        if key in reward_kwargs and isinstance(reward_kwargs[key], str):
            reward_kwargs[key] = _resolve_column_name(reward_kwargs[key])
    if "sigma_column" in reward_kwargs and isinstance(reward_kwargs["sigma_column"], str):
        # sigma_column can be either a param name (in model_params) or a column name in `data`
        if reward_kwargs["sigma_column"] not in model_params:
            reward_kwargs["sigma_column"] = _resolve_column_name(reward_kwargs["sigma_column"])

    # Check for missing columns and provide helpful error messages
    all_required_columns: set[str] = set()
    for col in state_accessor.state_columns:
        col_name = col[0] if isinstance(col, list) else col
        all_required_columns.add(str(col_name))
    for col in endogenous_accessor.state_columns:
        col_name = col[0] if isinstance(col, list) else col
        all_required_columns.add(str(col_name))
    for col in action_columns:
        all_required_columns.add(str(col))

    # Also check columns needed by reward function (unless it uses target_indices)
    if "target_indices" not in reward_kwargs:
        for key in (
            "target_column",
            "consumption_column",
            "labor_column",
            "consumption_young_column",
            "consumption_old_column",
            "sigma_column",
        ):
            if key in reward_kwargs and isinstance(reward_kwargs[key], str) and reward_kwargs[key] not in model_params:
                all_required_columns.add(reward_kwargs[key])

    missing_columns = all_required_columns - data_cols
    if missing_columns:
        available_columns = sorted(data.columns.tolist())
        original_cols = sorted(original_columns)
        raise KeyError(
            f"Missing required columns: {sorted(missing_columns)}\n"
            f"Available columns in {input_data_path} (after renaming): {available_columns}\n"
            f"Original columns (before renaming): {original_cols}\n"
            f"Column renames applied: {column_renames if column_renames else 'None'}\n"
            f"Required columns: {sorted(all_required_columns)}"
        )

    transitions = []

    current_discount_factor = 1.0
    accumulated_reward = 0.0

    # All reward functions now accept **kwargs, so we can pass all reward_kwargs directly
    data["REWARD_COMPUTED"] = reward_fn(data, model_params, **reward_kwargs)

    for idx, row in data.iterrows():
        if idx == 0:
            # first row is the initial state
            state_accessor.buffer.append(row)
            endogenous_accessor.buffer.append(row)
            continue

        state = state_accessor(row)
        endogenous = endogenous_accessor(row)
        action = row[action_columns].to_numpy()
        reward = float(row["REWARD_COMPUTED"])

        accumulated_reward += reward * current_discount_factor
        current_discount_factor *= discount_factor

        info_columns = list(set(row.index.to_list()) - set(state_accessor.get_columns() + action_columns))
        row_info = row[info_columns].to_dict()
        row_info["row_id"] = idx
        row_info["model_params"] = model_params
        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "endogenous": endogenous,
            "accumulated_reward": accumulated_reward,
            "truncated": False,
            "info": row_info,
        }

        transitions.append(transition)
    return pd.DataFrame(transitions)


def process_model_data(
    model_name: str,
    model_config: dict,
    model_params: dict,
    raw_data_path: Path,
    output_dir: Path,
) -> None:
    """Processes raw Dynare data for a specific model and configuration."""
    logger.info(f"Processing {model_name} with data from {raw_data_path}...")
    rl_env_conf = model_config["rl_env_settings"]

    def _parse_mod_symbol_tex_to_long(mod_text: str) -> dict[str, str]:
        """
        Parse Dynare .mod variable declarations to map possible CSV headers to long_name.

        Dynare's `M_.endo_names_tex` sometimes produces headers like `{c}` (SGU_2004),
        sometimes `$\\omega$`, etc. We map all of these to the variable's `long_name`.

        Only parses `var` and `varexo` declarations, not `parameters`, since CSV files
        only contain variable data.

        Returns mapping for keys:
          - sym (e.g. 'c')
          - '{sym}' (e.g. '{c}')
          - tex (e.g. '${c}$')
          - tex stripped of surrounding '$' (e.g. '{c}')
        """
        mapping: dict[str, str] = {}
        # Extract only var and varexo sections (CSV files don't contain parameters)
        var_section_pattern = re.compile(
            r"\b(?:var|varexo)\b.*?(?=\b(?:varexo|parameters|model|steady_state_model)\b|$)",
            re.DOTALL
        )
        var_sections = var_section_pattern.findall(mod_text)
        var_text = "\n".join(var_sections)

        # NOTE: this regex matches *actual* Dynare syntax in `.mod` files like:
        #   consumption ${C}$ (long_name='Consumption')
        # Keep the pattern readable: use real regex escapes (e.g. \b, \s) rather than
        # double-escaped sequences (\\b), otherwise it will never match.
        pat = re.compile(
            r"\b(?P<sym>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<tex>\$[^$]+\$)\s*\(long_name='(?P<long>[^']+)'\)"
        )
        for m in pat.finditer(var_text):
            sym = m.group("sym")
            tex = m.group("tex")
            long_name = m.group("long")
            # Symbol-based mappings always take precedence (add first)
            mapping[sym] = long_name
            mapping[sym.upper()] = long_name  # Add uppercase variant
            mapping[sym.lower()] = long_name  # Add lowercase variant
            mapping["{" + sym + "}"] = long_name
            mapping["{" + sym.upper() + "}"] = long_name  # Add uppercase variant
            mapping["{" + sym.lower() + "}"] = long_name  # Add lowercase variant
            # TeX mappings only if not already set (preserves symbol-based mappings)
            if tex not in mapping:
                mapping[tex] = long_name
            tex_stripped = tex.strip("$")
            if tex_stripped not in mapping:
                mapping[tex_stripped] = long_name
            if tex_stripped.upper() not in mapping:
                mapping[tex_stripped.upper()] = long_name
            if tex_stripped.lower() not in mapping:
                mapping[tex_stripped.lower()] = long_name
        return mapping

    def _build_column_renames_for_raw_csv(raw_csv_path: Path) -> dict[str, str]:
        """
        Build a renaming dict to make raw Dynare CSV column headers match our canonical names.
        Handles:
          1) TeX headers (${c}$) -> long_name via .mod parsing
          2) long_name variants -> canonical names via _COLUMN_ALIASES
        """
        renames: dict[str, str] = {}

        # 1) Header entries (symbol or TeX-like) -> long_name via .mod parsing.
        #
        # Read with pandas to get actual column names (pandas handles duplicate column renaming)
        try:
            # Read just the header to see what pandas does with duplicates
            df_header = pd.read_csv(raw_csv_path, nrows=0)
            header = list(df_header.columns)
        except Exception:
            # Fallback: read raw header line if pandas fails
            try:
                header_line = raw_csv_path.read_text(errors="ignore").splitlines()[0]
                header = [h.strip() for h in header_line.split(",") if h.strip()]
            except Exception:
                header = []

        mod_path = PathStorage().dynare_configs_root / f"{model_name}.mod"
        if mod_path.exists():
            sym_tex_to_long = _parse_mod_symbol_tex_to_long(mod_path.read_text(errors="ignore"))
            # Build set of all canonical names: all STATE_TOKENS (not just alias values)
            # This ensures long_names that are already in STATE_TOKENS are preserved
            canonical_names = set(Tokenizer.STATE_TOKENS)

            # Helper to resolve to canonical name, applying full alias chain
            def resolve_canonical(name: str) -> str:
                """Resolve to canonical name, applying full alias chain.

                If the name is already a canonical name (in canonical_names set),
                return it as-is without applying aliases.
                """
                # If name is already canonical, don't apply aliases
                if name in canonical_names:
                    return name
                seen = set()
                current = name
                while current in _COLUMN_ALIASES and current not in seen:
                    seen.add(current)
                    current = _COLUMN_ALIASES[current]
                return current

            # Build a mapping of TeX (stripped) to list of (symbol, long_name) for handling duplicates
            # This helps map pandas-renamed duplicates like {g}.1 to the correct variable
            tex_to_vars: dict[str, list[tuple[str, str]]] = {}
            mod_text = mod_path.read_text(errors="ignore")
            var_section_pattern = re.compile(
                r"\b(?:var|varexo)\b.*?(?=\b(?:varexo|parameters|model|steady_state_model)\b|$)",
                re.DOTALL
            )
            var_sections = var_section_pattern.findall(mod_text)
            var_text = "\n".join(var_sections)
            pat = re.compile(
                r"\b(?P<sym>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<tex>\$[^$]+\$)\s*\(long_name='(?P<long>[^']+)'\)"
            )
            for m in pat.finditer(var_text):
                sym = m.group("sym")
                tex = m.group("tex")
                long_name = m.group("long")
                tex_stripped = tex.strip("$")
                if tex_stripped not in tex_to_vars:
                    tex_to_vars[tex_stripped] = []
                tex_to_vars[tex_stripped].append((sym, long_name))

            # If we successfully parsed the header, map those entries.
            for c in header:
                # Check for pandas-renamed duplicates (e.g., {g}.1, {g}.2)
                base_col = c
                suffix_match = re.match(r"^(.+)\.(\d+)$", c)
                if suffix_match:
                    base_col = suffix_match.group(1)
                    suffix_num = int(suffix_match.group(2))
                else:
                    suffix_num = 0

                # Handle columns with pandas-renamed suffixes (.1, .2, etc.)
                if suffix_num > 0:
                    # Handle pandas-renamed duplicates: map {g}.1 to the second variable with same TeX
                    # Try to match base_col as TeX (with or without $)
                    tex_stripped = base_col.strip("$") if base_col.startswith("$") else base_col
                    if tex_stripped in tex_to_vars and suffix_num < len(tex_to_vars[tex_stripped]):
                        # Use the variable at index suffix_num (0-indexed, so .1 -> index 1)
                        sym, long_name = tex_to_vars[tex_stripped][suffix_num]
                        # Prefer long_name directly (most precise from model)
                        canonical_name = resolve_canonical(long_name)
                        if long_name in canonical_names:
                            target_name = long_name
                        elif canonical_name != long_name and canonical_name in canonical_names:
                            target_name = canonical_name
                        else:
                            target_name = long_name
                        if c != target_name:
                            # Only skip if c is already a canonical name (don't rename canonical -> something else)
                            is_current_canonical = c in canonical_names
                            if not is_current_canonical:
                                renames[c] = target_name
                elif base_col in sym_tex_to_long:
                    # Handle regular columns (no suffix)
                    long_name = sym_tex_to_long[base_col]
                    # Use long_name directly - it's the most precise name from the model definition
                    # Only apply alias resolution if long_name itself needs canonicalization
                    # (e.g., "Total Factor Productivity" -> "LoggedProductivity")
                    # But prefer the exact long_name if it's already precise
                    canonical_name = resolve_canonical(long_name)

                    # IMPORTANT: Prefer long_name if it's already a canonical name
                    # This preserves the precise meaning from the model definition
                    # Only apply alias resolution if long_name is NOT canonical
                    if long_name in canonical_names:
                        # long_name is already canonical, use it directly (most precise)
                        target_name = long_name
                    elif canonical_name in canonical_names:
                        # long_name resolved to a canonical name via aliases, use it
                        target_name = canonical_name
                    else:
                        # Neither is canonical, use long_name as-is (most precise from model)
                        target_name = long_name

                    # Only rename if:
                    # 1. Column name differs from target name
                    # 2. Current column is NOT already a canonical name (prevent canonical -> something else)
                    if c != target_name:
                        # Check if current column is already canonical
                        is_current_canonical = c in canonical_names or c not in _COLUMN_ALIASES

                        # Never rename a canonical name to something else
                        # Only rename non-canonical aliases to their canonical form
                        if not is_current_canonical:
                            renames[c] = target_name

        # 2) long_name variants -> canonical names
        # Only add aliases for columns that actually exist in the CSV header
        # This prevents adding renames that would rename canonical names to non-canonical ones
        # IMPORTANT: Don't override renames we already set from .mod file parsing (step 1)
        header_set = set(header)
        for alias_key, canonical_value in _COLUMN_ALIASES.items():
            # Only add rename if:
            # - The alias key exists in the header
            # - It's not already canonical
            # - We haven't already set a rename for this column (from .mod file parsing)
            if alias_key in header_set and alias_key != canonical_value and alias_key not in renames:
                renames[alias_key] = canonical_value
        return renames

    def _resolve_discount_factor(params: dict, rl_conf: dict) -> float:
        """
        Resolve discount factor used for accumulated reward computation.

        Historically we assumed Dynare parameter name `beta`, but different .mod files use
        different conventions (e.g. `BETTA`, `discount_factor`).

        Priority:
          1) Explicit constant: rl_env_settings.discount_factor
          2) Explicit param name: rl_env_settings.discount_factor_param
          3) Auto-detect from params keys (case/underscore-insensitive)
          4) Fallback to 1.0 (no discounting)
        """
        if isinstance(rl_conf, dict):
            if "discount_factor" in rl_conf and rl_conf["discount_factor"] is not None:
                return float(rl_conf["discount_factor"])

            if "discount_factor_param" in rl_conf and rl_conf["discount_factor_param"] is not None:
                k = str(rl_conf["discount_factor_param"])
                if k not in params:
                    raise KeyError(
                        f"discount_factor_param='{k}' not found in model_params keys={list(params.keys())}"
                    )
                return float(params[k])

        def norm_key(s: str) -> str:
            return "".join(ch for ch in s.lower() if ch.isalnum())

        norm_map = {norm_key(str(k)): k for k in params.keys()}
        for candidate in ("beta", "betta", "discountfactor", "discount_factor"):
            nk = norm_key(candidate)
            if nk in norm_map:
                return float(params[norm_map[nk]])

        logger.warning(
            f"[{model_name}] Could not resolve discount factor from params; "
            f"known keys={list(params.keys())}. Falling back to 1.0"
        )
        return 1.0

    # Extract configuration number from the filename (if any)
    config_match = re.search(r"_config_(\d+)_raw\.csv$", str(raw_data_path))
    config_suffix = f"_config_{config_match.group(1)}" if config_match else ""

    # Generate output path
    output_path = Path(output_dir) / f"{model_name}{config_suffix}.parquet"

    state_accessor = StateAccessor(rl_env_conf["input"]["state_columns"])
    endogenous_columns = rl_env_conf["input"].get("endogenous_columns", [])
    endogenous_accessor = StateAccessor(endogenous_columns)

    reward_fn = get_reward_object(rl_env_conf["reward"])
    mod_file_path = PathStorage().dynare_configs_root / f"{model_name}.mod"
    transitions = dynare_trajectories2rl_transitions(
        input_data_path=raw_data_path,
        state_accessor=state_accessor,
        endogenous_accessor=endogenous_accessor,
        action_columns=rl_env_conf["input"]["action_columns"],
        reward_fn=reward_fn, # type: ignore
        reward_kwargs=rl_env_conf["reward_kwargs"],
        discount_factor=_resolve_discount_factor(model_params, rl_env_conf),
        model_params=model_params,
        column_renames=_build_column_renames_for_raw_csv(raw_data_path),
        mod_file_path=mod_file_path,
    )
    logger.info("Transitions successfully generated.")

    logger.info("Saving data...")

    # Persist the economics model identifier through the pipeline so downstream
    # dataset builders can group episodes robustly even if filenames include hashes.
    if "info" not in transitions.columns:
        logger.warning(
            f"[{model_name}] Transitions missing 'info' column (len={len(transitions)}). "
            "Creating an empty one."
        )
        transitions["info"] = pd.Series([{}] * len(transitions))

    def _add_env_group(x: object) -> dict:
        if isinstance(x, dict):
            base = x
        else:
            base = {}
        return base | {"env_group": model_name}

    transitions["info"] = transitions["info"].apply(_add_env_group)

    # Map column names to long_names from .mod file for precise descriptions
    def _map_columns_to_long_names(columns: list[str]) -> list[str]:
        """Map column names to their long_name from .mod file, falling back to column name if not found."""
        if not mod_file_path.exists():
            return columns  # Fallback to column names if .mod file not available

        mod_text = mod_file_path.read_text(errors="ignore")
        sym_tex_to_long = _parse_mod_symbol_tex_to_long(mod_text)

        # Create reverse mapping: long_name -> symbol (to find which symbol has this long_name)
        long_name_to_symbol: dict[str, str] = {}
        var_section_pattern = re.compile(
            r"\b(?:var|varexo)\b.*?(?=\b(?:varexo|parameters|model|steady_state_model)\b|$)",
            re.DOTALL
        )
        var_sections = var_section_pattern.findall(mod_text)
        var_text = "\n".join(var_sections)
        pat = re.compile(
            r"\b(?P<sym>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<tex>\$[^$]+\$)\s*\(long_name='(?P<long>[^']+)'\)"
        )
        for m in pat.finditer(var_text):
            sym = m.group("sym")
            long_name = m.group("long")
            # Store first occurrence (prefer lowercase symbol)
            if long_name not in long_name_to_symbol or sym.islower():
                long_name_to_symbol[long_name] = sym

        # Collect all long_names from .mod file for direct matching
        all_long_names = set(sym_tex_to_long.values())

        long_names = []
        for col in columns:
            # Strategy 1: Column is already a long_name from .mod file
            if col in all_long_names:
                long_names.append(col)
            # Strategy 2: Column name (as symbol/TeX) maps directly to a long_name
            elif col in sym_tex_to_long:
                long_names.append(sym_tex_to_long[col])
            # Strategy 3: Resolve through alias chain and check if canonical name is a long_name
            else:
                canonical = col
                seen = set()
                while canonical in _COLUMN_ALIASES and canonical not in seen:
                    seen.add(canonical)
                    canonical = _COLUMN_ALIASES[canonical]

                # Check if canonical name is already a long_name
                if canonical in all_long_names:
                    long_names.append(canonical)
                # Check if canonical name maps to a long_name via symbol lookup
                elif canonical in sym_tex_to_long:
                    long_names.append(sym_tex_to_long[canonical])
                else:
                    # Fallback: use column name as-is (might be a canonical name not in .mod)
                    long_names.append(col)

        return long_names

    # Use long_names from .mod files for descriptions
    state_columns = state_accessor.get_columns()
    action_columns = rl_env_conf["input"]["action_columns"]
    endogenous_columns = endogenous_accessor.get_columns()

    transitions["action_description"] = pd.Series([_map_columns_to_long_names(action_columns)] * len(transitions))
    transitions["state_description"] = pd.Series([_map_columns_to_long_names(state_columns)] * len(transitions))
    transitions["endogenous_description"] = pd.Series([_map_columns_to_long_names(endogenous_columns)] * len(transitions))

    transitions.to_parquet(output_path)

    logger.info(f"Data saved to {output_path}")


def extract_model_name(filename: str) -> str:
    """Extracts the base model name from a filename.

    Args:
        filename (str): The filename, e.g., "Born_Pfeifer_2018_MP_config_1_raw.csv".

    Returns:
        str: The base model name, e.g., "Born_Pfeifer_2018_MP".
    """
    # Удаляем суффикс "_config_*_raw.csv"
    if "_config_" in filename and "_raw" in filename:
        return filename.split("_config_")[0]
    # Если суффикса нет, возвращаем имя файла без расширения
    return filename.replace("_raw", "")


@hydra.main(config_path="../dynare/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    config = cast(dict, OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    path_storage = PathStorage(data_folder=config["metadata"]["data_folder"])
    path_storage.raw_root.mkdir(parents=True, exist_ok=True)
    path_storage.processed_root.mkdir(parents=True, exist_ok=True)
    logger.info("Running models...")
    output_files = run_models(config["models"], path_storage.raw_root)
    logger.info("Models run successfully.")

    # Ensure output directory exists
    os.makedirs(path_storage.raw_root, exist_ok=True)
    os.makedirs(path_storage.processed_root, exist_ok=True)
    for raw_data_file, params_file in output_files:
        with open(params_file, 'r') as f:
            model_params = yaml.load(f, Loader=yaml.FullLoader)

        model_name = extract_model_name(raw_data_file.stem)

        if model_name not in config["models"]:
            logger.warning(f"Model {model_name} not found in config")
            continue

        try:
            process_model_data(
                model_name=model_name,
                model_config=config["models"][model_name],
                model_params=model_params,
                raw_data_path=raw_data_file,
                output_dir=path_storage.processed_root,
            )
        except Exception as e:
            logger.error(f"Error processing {model_name}: {e}")
            logger.error(traceback.format_exc())
            continue


if __name__ == "__main__":
    main()

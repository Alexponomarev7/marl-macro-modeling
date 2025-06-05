import pytest
from loguru import logger
from research.data.utils import GenerationType, generate_model
import numpy as np


@pytest.mark.parametrize("parameters", [{"start_capital": 1.0, "delta": 0.01, "alpha": 0.33, "beta": 0.95}])
def test_ramsey_env(parameters):
    df_dynare, params1 = generate_model(
        "Ramsey", parameters, periods=50, type=GenerationType.DYNARE
    )
    df_gymnasium, params2 = generate_model(
        "Ramsey", parameters, periods=50, type=GenerationType.GYMNASIUM
    )
    df_gymnasium_with_trajectory, params3 = generate_model(
        "Ramsey", parameters, periods=50, type=GenerationType.GYMNASIUM, trajectory=df_dynare
    )

    df_dynare = df_dynare.iloc[:25]
    df_gymnasium = df_gymnasium.iloc[:25]
    df_gymnasium_with_trajectory = df_gymnasium_with_trajectory.iloc[:25]

    assert np.allclose(df_dynare, df_gymnasium, atol=5e-2), "dataframes are not the same"
    assert params1 == params2, "parameters should be the same"

    assert np.allclose(df_dynare, df_gymnasium_with_trajectory, atol=1e-3), "dataframes are not the same"
    assert params1 == params3, "parameters should be the same"

if __name__ == "__main__":
    pytest.main()

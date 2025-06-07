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

    params1.pop("k_ss")
    params1.pop("c_ss")

    assert np.allclose(df_dynare, df_gymnasium, atol=5e-2), "dataframes are not the same"
    assert params1 == params2, "parameters should be the same"

    assert np.allclose(df_dynare, df_gymnasium_with_trajectory, atol=1e-3), "dataframes are not the same"
    assert params1 == params3, "parameters should be the same"

@pytest.mark.parametrize("parameters", [{
    "alpha": 0.32, "beta": 0.98**4, "delta": 1.03**4-1,
    "gamma_a": 2.0, "omega": 1.6, "theta": 1.4 * 1.6, 
    "phi": 4.81, "psi": 2.87, "dbar": 0.007, "gbar": 1.01,
    "s_share": 0.10, "rho_a": 0.86, "rho_g": 0.32,
    "rho_nu": 0.85, "rho_mu": 0.91, "rho_s": 0.21,
    "start_capital": 0.1
}])
def test_garcia_cicco_env(parameters):
    columns = [
        # state
        "Capital",
        "LoggedProductivity",
        "Debt",
        "Output",
        "PreferenceShock",
        "CountryPremiumShock",
        "InterestRate",
        "TechGrowthRate",
        "TradeBalance",
        "Investment",
        # action
        "Consumption",
        "HoursWorked",
        # info
        # "MUConsumption"
    ]
    
    df_dynare, params1 = generate_model(
        "GarciaCicco_et_al_2010", parameters, periods=50, type=GenerationType.DYNARE
    )
    df_gymnasium, params2 = generate_model(
        "GarciaCicco_et_al_2010", params1, periods=50, type=GenerationType.GYMNASIUM, trajectory=df_dynare
    )

    params1.pop("k_ss")
    params1.pop("c_ss")

    assert params1 == params2, "parameters should be the same"

    df_dynare = df_dynare[columns].iloc[1:50]
    df_gymnasium = df_gymnasium[columns].iloc[1:50]

    assert np.allclose(df_dynare[["Capital", "Output"]], df_gymnasium[["Capital", "Output"]], atol=1e-3), "dataframes are not the same"

if __name__ == "__main__":
    pytest.main()

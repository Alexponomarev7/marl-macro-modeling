% Caldara, Fernandez-Villaverde, Rubio-Ramirez, Yao (2012)
% Computing DSGE Models with Recursive Preferences and Stochastic Volatility
%
% Features:
%   - Epstein-Zin recursive preferences
%   - Stochastic volatility in technology shocks
%   - Second-order perturbation solution required
%
% Key parameters:
%   - gamma_risk: risk aversion (avoid reserved word 'gamma')
%   - eis: elasticity of intertemporal substitution
%   - Extreme calibration: gamma_risk=40, eis=0.5 (high risk aversion, low EIS)
%   - Moderate calibration: gamma_risk=5, eis=0.5
%
% Timing convention:
%   - Capital_t is end-of-period t capital
%   - Production at t uses Capital_{t-1}

var ValueFunction           $ValueFunction$           (long_name='Value Function')
    Output                  $Output$                  (long_name='Output')
    Consumption             $Consumption$             (long_name='Consumption')
    Capital                 $Capital$                 (long_name='Capital')
    Investment              $Investment$              (long_name='Investment')
    Labor                   $Labor$                   (long_name='Labor')
    LoggedProductivity      $LoggedProductivity$      (long_name='Technology Shock')
    ValueAuxiliary          $ValueAuxiliary$          (long_name='Auxiliary Variable For Value Function')
    ExpectedSDF             $ExpectedSDF$             (long_name='Expected Stochastic Discount Factor')
    LoggedVolatility        $LoggedVolatility$        (long_name='Log Volatility')
    ExpectedReturnCapital   $ExpectedReturnCapital$   (long_name='Expected Return On Capital')
    RiskFreeRate            $RiskFreeRate$            (long_name='Risk-Free Rate');

varexo TechnologyInnovation   $TechnologyInnovation$   (long_name='Technology Shock Innovation')
       VolatilityInnovation   $VolatilityInnovation$   (long_name='Volatility Shock');

parameters beta gamma_risk delta_depreciation nu_consumption eis
           rho_technology zeta_capital rho_volatility sigma_bar eta_volatility
           l_ss k_ss c_ss y_ss i_ss v_ss s_ss r_k_ss r_f_ss;

@#if !defined(beta)
    @#define beta = 0.991
@#endif

@#if !defined(zeta_capital)
    @#define zeta_capital = 0.3
@#endif

@#if !defined(delta_depreciation)
    @#define delta_depreciation = 0.0196
@#endif

@#if !defined(rho_technology)
    @#define rho_technology = 0.95
@#endif

@#if !defined(eis)
    @#define eis = 0.5
@#endif

@#if !defined(gamma_risk)
    @#define gamma_risk = 5.0
@#endif

@#if !defined(rho_volatility)
    @#define rho_volatility = 0.9
@#endif

@#if !defined(sigma_bar)
    @#define sigma_bar = -3.86
@#endif

@#if !defined(eta_volatility)
    @#define eta_volatility = 0.06
@#endif

@#if !defined(technology_shock_stderr)
    @#define technology_shock_stderr = 1.0
@#endif

@#if !defined(volatility_shock_stderr)
    @#define volatility_shock_stderr = 1.0
@#endif

beta = @{beta};
zeta_capital = @{zeta_capital};
delta_depreciation = @{delta_depreciation};
rho_technology = @{rho_technology};
eis = @{eis};
gamma_risk = @{gamma_risk};
rho_volatility = @{rho_volatility};
sigma_bar = @{sigma_bar};
eta_volatility = @{eta_volatility};

l_ss = 1 / 3;
k_ss = ((1 - beta * (1 - delta_depreciation)) / (zeta_capital * beta))^(1 / (zeta_capital - 1)) * l_ss;
c_ss = k_ss^zeta_capital * l_ss^(1 - zeta_capital) - delta_depreciation * k_ss;
nu_consumption = c_ss / ((1 - zeta_capital) * k_ss^zeta_capital * l_ss^(-zeta_capital) * (1 - l_ss) + c_ss);

i_ss = delta_depreciation * k_ss;
y_ss = k_ss^zeta_capital * l_ss^(1 - zeta_capital);
v_ss = c_ss^nu_consumption * (1 - l_ss)^(1 - nu_consumption);
s_ss = v_ss^(1 - gamma_risk);
r_k_ss = zeta_capital * k_ss^(zeta_capital - 1) * l_ss^(1 - zeta_capital) - delta_depreciation;
r_f_ss = 1 / beta - 1;

model;

% Composite parameter for Epstein-Zin preferences
#theta_ez = (1 - gamma_risk) / (1 - (1 / eis));

[name='Value Function (Epstein-Zin recursive utility)']
ValueFunction = ((1 - beta) * ((Consumption^nu_consumption) * ((1 - Labor)^(1 - nu_consumption)))^((1 - gamma_risk) / theta_ez) 
                + beta * ValueAuxiliary^(1 / theta_ez))^(theta_ez / (1 - gamma_risk));

[name='Auxiliary Variable for Continuation Value']
ValueAuxiliary = ValueFunction(+1)^(1 - gamma_risk);

[name='Euler Equation (Epstein-Zin)']
1 = beta * (((1 - Labor(+1)) / (1 - Labor))^(1 - nu_consumption) * (Consumption(+1) / Consumption)^nu_consumption)^((1 - gamma_risk) / theta_ez) 
    * Consumption / Consumption(+1)
    * ((ValueFunction(+1)^(1 - gamma_risk)) / ValueAuxiliary)^(1 - (1 / theta_ez)) 
    * (zeta_capital * exp(LoggedProductivity(+1)) * Capital^(zeta_capital - 1) * Labor(+1)^(1 - zeta_capital) + 1 - delta_depreciation);

[name='Expected Return on Capital']
ExpectedReturnCapital = zeta_capital * exp(LoggedProductivity(+1)) * Capital^(zeta_capital - 1) * Labor(+1)^(1 - zeta_capital) - delta_depreciation;

[name='Expected Stochastic Discount Factor']
ExpectedSDF = beta * (((1 - Labor(+1)) / (1 - Labor))^(1 - nu_consumption) * (Consumption(+1) / Consumption)^nu_consumption)^((1 - gamma_risk) / theta_ez) 
            * Consumption / Consumption(+1)
            * ((ValueFunction(+1)^(1 - gamma_risk)) / ValueAuxiliary)^(1 - (1 / theta_ez));

[name='Risk-Free Rate']
RiskFreeRate = (1 / ExpectedSDF - 1);

[name='Labor Supply FOC']
((1 - nu_consumption) / nu_consumption) * (Consumption / (1 - Labor)) = (1 - zeta_capital) * exp(LoggedProductivity) * (Capital(-1)^zeta_capital) * (Labor^(-zeta_capital));

[name='Budget Constraint']
Consumption + Investment = exp(LoggedProductivity) * (Capital(-1)^zeta_capital) * (Labor^(1 - zeta_capital));

[name='Capital Accumulation']
Capital = (1 - delta_depreciation) * Capital(-1) + Investment;

[name='Technology Shock Process with Stochastic Volatility']
LoggedProductivity = rho_technology * LoggedProductivity(-1) + exp(LoggedVolatility) * TechnologyInnovation;

[name='Output Definition']
Output = exp(LoggedProductivity) * (Capital(-1)^zeta_capital) * (Labor^(1 - zeta_capital));

[name='Volatility Process']
LoggedVolatility = (1 - rho_volatility) * sigma_bar + rho_volatility * LoggedVolatility(-1) + eta_volatility * VolatilityInnovation;

end;

steady_state_model;
    Labor = l_ss;
    Capital = k_ss;
    Consumption = c_ss;
    Investment = i_ss;
    Output = y_ss;
    ValueFunction = v_ss;
    ValueAuxiliary = s_ss;
    LoggedProductivity = 0;
    LoggedVolatility = sigma_bar;
    ExpectedSDF = beta;
    ExpectedReturnCapital = r_k_ss;
    RiskFreeRate = r_f_ss;
end;

shocks;
    var TechnologyInnovation;
    stderr @{technology_shock_stderr};
    
    var VolatilityInnovation;
    stderr @{volatility_shock_stderr};
end;

steady;
check;

stoch_simul(order=2, periods=@{periods}, drop=0, irf=0, nomoments, nofunctions, nograph, nocorr, noprint);

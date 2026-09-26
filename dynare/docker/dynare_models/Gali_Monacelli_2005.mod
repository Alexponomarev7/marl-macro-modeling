/*
 * Gali, Jordi and Tommaso Monacelli (2005): "Monetary Policy and Exchange Rate Volatility in a
 * Small Open Economy", Review of Economic Studies 72: 707-734, CPI-inflation Taylor rule regime.
 * Adapted from Johannes Pfeifer's DSGE_mod implementation (GPL-3.0); price and exchange-rate levels
 * are dropped. Variables are percent deviations from steady state.
 */

@#ifndef periods
    @#define periods = 1000
@#endif

var InflationH                      $InflationH$                      (long_name='Domestic inflation')
    OutputGap                       $OutputGap$                       (long_name='Output Gap')
    OutputDeviationFromSteadyState  $OutputDeviationFromSteadyState$  (long_name='Output Deviation From Steady State')
    NaturalOutput                   $NaturalOutput$                   (long_name='Natural Output')
    NaturalInterestRate             $NaturalInterestRate$             (long_name='Natural Interest Rate')
    NominalInterestRate             $NominalInterestRate$             (long_name='Nominal Interest Rate')
    TermsOfTrade                    $TermsOfTrade$                    (long_name='Terms of trade')
    InflationCPI                    $InflationCPI$                    (long_name='CPI inflation')
    NominalDepreciationRate         $NominalDepreciationRate$         (long_name='Nominal Depreciation Rate')
    OutputForeign                   $OutputForeign$                   (long_name='World output')
    Employment                      $Employment$                      (long_name='Employment')
    NetExports                      $NetExports$                      (long_name='Net Exports')
    RealWage                        $RealWage$                        (long_name='Real Wage')
    TechnologyShock                 $TechnologyShock$                 (long_name='Technology Shock')
    ConsumptionH                    $ConsumptionH$                    (long_name='Domestic consumption');

varexo WorldOutputInnovation  $WorldOutputInnovation$  (long_name='World output shock')
       TechnologyInnovation   $TechnologyInnovation$   (long_name='Technology shock');

parameters sigma    $\sigma$      (long_name='risk aversion')
           eta      $\eta$        (long_name='substitution home foreign')
           gamma_f  $\gamma$      (long_name='substitution between foreign')
           phi      $\varphi$     (long_name='inverse Frisch elasticity')
           epsilon  $\varepsilon$ (long_name='elasticity of substitution between varieties')
           theta    $\theta$      (long_name='Calvo parameter')
           beta     $\beta$       (long_name='discount factor')
           alpha    $\alpha$      (long_name='openness')
           phi_pi   $\phi_\pi$    (long_name='feedback Taylor rule inflation')
           rhoa     $\rho_a$      (long_name='autocorrelation TFP')
           rhoy     $\rho_y$      (long_name='autocorrelation foreign output');

% Calibration: Gali & Monacelli (2005), Section 6 (p. 723)
@#if !defined(sigma)
    @#define sigma = 1
@#endif
@#if !defined(eta)
    @#define eta = 1
@#endif
@#if !defined(gamma_f)
    @#define gamma_f = 1
@#endif
@#if !defined(phi)
    @#define phi = 3
@#endif
@#if !defined(epsilon)
    @#define epsilon = 6
@#endif
@#if !defined(theta)
    @#define theta = 0.75
@#endif
@#if !defined(beta)
    @#define beta = 0.99
@#endif
@#if !defined(alpha)
    @#define alpha = 0.4
@#endif
@#if !defined(phi_pi)
    @#define phi_pi = 1.5
@#endif
@#if !defined(rhoa)
    @#define rhoa = 0.66
@#endif
@#if !defined(rhoy)
    @#define rhoy = 0.86
@#endif
% Shock scales in percent: sigma_a = 0.0071, sigma_y* = 0.0078 (p. 723)
@#if !defined(technology_shock_stderr)
    @#define technology_shock_stderr = 0.71
@#endif
@#if !defined(world_output_shock_stderr)
    @#define world_output_shock_stderr = 0.78
@#endif

sigma = @{sigma};
eta = @{eta};
gamma_f = @{gamma_f};
phi = @{phi};
epsilon = @{epsilon};
theta = @{theta};
beta = @{beta};
alpha = @{alpha};
phi_pi = @{phi_pi};
rhoa = @{rhoa};
rhoy = @{rhoy};

model(linear);
#omega = sigma * gamma_f + (1 - alpha) * (sigma * eta - 1);
#sigma_a = sigma / ((1 - alpha) + alpha * omega);
#Theta = (sigma * gamma_f - 1) + (1 - alpha) * (sigma * eta - 1);
#lambda = (1 - (beta * theta)) * (1 - theta) / theta;
#kappa_a = lambda * (sigma_a + phi);
#Gamma = (1 + phi) / (sigma_a + phi);
#Psi = -Theta * sigma_a / (sigma_a + phi);
[name='Equation (37), IS Curve']
OutputGap = OutputGap(+1) - sigma_a^(-1) * (NominalInterestRate - InflationH(+1) - NaturalInterestRate);
[name='Equation (36), Phillips Curve']
InflationH = beta * InflationH(+1) + kappa_a * OutputGap;
[name='Equation below (37), natural rate']
NaturalInterestRate = -sigma_a * Gamma * (1 - rhoa) * TechnologyShock + alpha * sigma_a * (Theta + Psi) * (OutputForeign(+1) - OutputForeign);
[name='Equation (35), natural output']
NaturalOutput = Gamma * TechnologyShock + alpha * Psi * OutputForeign;
[name='Output gap']
OutputGap = OutputDeviationFromSteadyState - NaturalOutput;
[name='Equation (29)']
OutputDeviationFromSteadyState = OutputForeign + sigma_a^(-1) * TermsOfTrade;
[name='Equation (14), CPI inflation']
InflationCPI = InflationH + alpha * (TermsOfTrade - TermsOfTrade(-1));
[name='Equation (15) in first differences, constant world prices (pi* = 0)']
TermsOfTrade = TermsOfTrade(-1) + NominalDepreciationRate - InflationH;
[name='Equation (22), employment']
OutputDeviationFromSteadyState = TechnologyShock + Employment;
[name='Equation (31), net exports']
NetExports = alpha * (omega / sigma - 1) * TermsOfTrade;
[name='Equation (27), domestic consumption']
OutputDeviationFromSteadyState = ConsumptionH + alpha * omega / sigma * TermsOfTrade;
[name='Real wage (labor supply)']
RealWage = sigma * ConsumptionH + phi * Employment;
[name='Technology process, p. 723']
TechnologyShock = rhoa * TechnologyShock(-1) + TechnologyInnovation;
[name='World output process, p. 723']
OutputForeign = rhoy * OutputForeign(-1) + WorldOutputInnovation;
[name='CPI inflation-based Taylor rule (CITR)']
NominalInterestRate = phi_pi * InflationCPI;
end;

shocks;
    var TechnologyInnovation; stderr @{technology_shock_stderr};
    var WorldOutputInnovation; stderr @{world_output_shock_stderr};
end;

steady;
check;

stoch_simul(order=1, periods=@{periods}, irf=0, nomoments, nofunctions, nograph, nocorr, noprint);

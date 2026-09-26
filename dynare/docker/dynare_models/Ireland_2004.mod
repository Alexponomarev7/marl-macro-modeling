/*
 * New Keynesian model of Peter Ireland (2004): Technology shocks in the New Keynesian model,
 * Review of Economics and Statistics, 86(4), pp. 923-936. Adapted from Johannes Pfeifer's DSGE_mod
 * implementation (GPL-3.0). Linear, in percent deviations from trend; the rule is in differences:
 *   r_t - r_{t-1} = rho_pi pi_t + rho_g g_t + rho_x x_t + eps_r
 */

@#ifndef periods
    @#define periods = 1000
@#endif

var PreferenceShock                  $PreferenceShock$                  (long_name='Preference Shock')
    CostPushShock                    $CostPushShock$                    (long_name='Cost Push Shock')
    TechnologyShock                  $TechnologyShock$                  (long_name='Technology Shock')
    OutputGap                        $OutputGap$                        (long_name='Output Gap')
    PriceInflation                   $PriceInflation$                   (long_name='Price Inflation')
    OutputDeviationFromSteadyState   $OutputDeviationFromSteadyState$   (long_name='Output Deviation From Steady State')
    OutputGrowth                     $OutputGrowth$                     (long_name='Output Growth')
    NominalInterestRate              $NominalInterestRate$              (long_name='Nominal Interest Rate')
    AnnualNominalRate                $AnnualNominalRate$                (long_name='Annualized Nominal Interest Rate')
    AnnualInflation                  $AnnualInflation$                  (long_name='Annualized Inflation Rate');

varexo PreferenceInnovation  $PreferenceInnovation$  (long_name='Preference Shock')
       CostPushInnovation    $CostPushInnovation$    (long_name='Cost Push Shock')
       TechnologyInnovation  $TechnologyInnovation$  (long_name='Technology Shock')
       MonetaryInnovation    $MonetaryInnovation$    (long_name='Monetary Policy Shock');

parameters beta      ${\beta}$       (long_name='discount factor')
           alpha_x   ${\alpha_x}$    (long_name='backward-looking share in the IS curve')
           alpha_pi  ${\alpha_\pi}$  (long_name='backward-looking share in the Phillips curve')
           rho_a     ${\rho_a}$      (long_name='autocorrelation preference shock')
           rho_e     ${\rho_e}$      (long_name='autocorrelation cost-push shock')
           omega     ${\omega}$      (long_name='weight of the preference shock in the output gap')
           psi       ${\psi}$        (long_name='slope of the Phillips curve')
           rho_pi    ${\rho_\pi}$    (long_name='inflation feedback')
           rho_g     ${\rho_g}$      (long_name='output growth feedback')
           rho_x     ${\rho_x}$      (long_name='output gap feedback');

% Calibration: beta and psi as in Ireland (2004); the rest post-1980 estimates (Table 1)
@#if !defined(beta)
    @#define beta = 0.99
@#endif
@#if !defined(psi)
    @#define psi = 0.1
@#endif
@#if !defined(omega)
    @#define omega = 0.0581
@#endif
@#if !defined(alpha_x)
    @#define alpha_x = 0.00001
@#endif
@#if !defined(alpha_pi)
    @#define alpha_pi = 0.00001
@#endif
@#if !defined(rho_pi)
    @#define rho_pi = 0.3866
@#endif
@#if !defined(rho_g)
    @#define rho_g = 0.3960
@#endif
@#if !defined(rho_x)
    @#define rho_x = 0.1654
@#endif
@#if !defined(rho_a)
    @#define rho_a = 0.9048
@#endif
@#if !defined(rho_e)
    @#define rho_e = 0.9907
@#endif
% innovation standard deviations in percent
@#if !defined(preference_shock_stderr)
    @#define preference_shock_stderr = 3.02
@#endif
@#if !defined(cost_push_shock_stderr)
    @#define cost_push_shock_stderr = 0.02
@#endif
@#if !defined(technology_shock_stderr)
    @#define technology_shock_stderr = 0.89
@#endif
@#if !defined(monetary_shock_stderr)
    @#define monetary_shock_stderr = 0.28
@#endif

beta = @{beta};
psi = @{psi};
omega = @{omega};
alpha_x = @{alpha_x};
alpha_pi = @{alpha_pi};
rho_pi = @{rho_pi};
rho_g = @{rho_g};
rho_x = @{rho_x};
rho_a = @{rho_a};
rho_e = @{rho_e};

model(linear);
[name='temporary preference shock (15)']
PreferenceShock = rho_a * PreferenceShock(-1) + PreferenceInnovation;
[name='temporary cost-push shock (16)']
CostPushShock = rho_e * CostPushShock(-1) + CostPushInnovation;
[name='technology shock (17)']
TechnologyShock = TechnologyInnovation;
[name='New Keynesian IS curve (23)']
OutputGap = alpha_x * OutputGap(-1) + (1 - alpha_x) * OutputGap(+1)
    - (NominalInterestRate - PriceInflation(+1)) + (1 - omega) * (1 - rho_a) * PreferenceShock;
[name='New Keynesian Phillips curve (24)']
PriceInflation = beta * (alpha_pi * PriceInflation(-1) + (1 - alpha_pi) * PriceInflation(+1))
    + psi * OutputGap - CostPushShock;
[name='output gap (20)']
OutputGap = OutputDeviationFromSteadyState - omega * PreferenceShock;
[name='growth rate of output (21)']
OutputGrowth = OutputDeviationFromSteadyState - OutputDeviationFromSteadyState(-1) + TechnologyShock;
[name='policy rule (22)']
NominalInterestRate - NominalInterestRate(-1) = rho_pi * PriceInflation + rho_g * OutputGrowth
    + rho_x * OutputGap + MonetaryInnovation;
[name='annualized interest rate']
AnnualNominalRate = 4 * NominalInterestRate;
[name='annualized inflation rate']
AnnualInflation = 4 * PriceInflation;
end;

shocks;
    var PreferenceInnovation = @{preference_shock_stderr}^2;
    var CostPushInnovation = @{cost_push_shock_stderr}^2;
    var TechnologyInnovation = @{technology_shock_stderr}^2;
    var MonetaryInnovation = @{monetary_shock_stderr}^2;
end;

steady;
check;

stoch_simul(order=1, periods=@{periods}, irf=0, nomoments, nofunctions, nograph, nocorr, noprint);

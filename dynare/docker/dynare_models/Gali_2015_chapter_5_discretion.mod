/*
 * Optimal monetary policy under discretion in the basic New Keynesian model of Jordi Gali (2015):
 * Monetary Policy, Inflation, and the Business Cycle, 2nd ed., Chapter 5. The nominal rate
 * minimizes E sum beta^t (pi_t^2 + vartheta x_t^2). Adapted from Johannes Pfeifer's DSGE_mod
 * implementation (GPL-3.0). Variables are percent deviations from the efficient steady state.
 */

@#ifndef periods
    @#define periods = 1000
@#endif

var PriceInflation                  $PriceInflation$                  (long_name='Price Inflation')
    OutputGap                       $OutputGap$                       (long_name='Output Gap')
    NaturalOutput                   $NaturalOutput$                   (long_name='Natural Output')
    OutputDeviationFromSteadyState  $OutputDeviationFromSteadyState$  (long_name='Output Deviation From Steady State')
    EfficientInterestRate           $EfficientInterestRate$           (long_name='Efficient Interest Rate')
    EfficientOutput                 $EfficientOutput$                 (long_name='Efficient Output')
    WelfareRelevantOutputGap        $WelfareRelevantOutputGap$        (long_name='Welfare Relevant Output Gap')
    RealInterestRate                $RealInterestRate$                (long_name='Real Interest Rate')
    NominalInterestRate             $NominalInterestRate$             (long_name='Nominal Interest Rate')
    HoursWorked                     $HoursWorked$                     (long_name='Hours Worked')
    MoneyGrowthAnnual               $MoneyGrowthAnnual$               (long_name='Money Growth Annualized')
    CostPushShock                   $CostPushShock$                   (long_name='Cost Push Shock')
    TechnologyShock                 $TechnologyShock$                 (long_name='Technology Shock')
    PreferenceShock                 $PreferenceShock$                 (long_name='Preference Shock')
    AnnualRealRate                  $AnnualRealRate$                  (long_name='Annualized Real Interest Rate')
    AnnualNominalRate               $AnnualNominalRate$               (long_name='Annualized Nominal Interest Rate')
    AnnualInflation                 $AnnualInflation$                 (long_name='Annualized Inflation Rate');

varexo TechnologyInnovation  $TechnologyInnovation$  (long_name='Technology Shock')
       CostPushInnovation    $CostPushInnovation$    (long_name='Cost Push Shock')
       PreferenceInnovation  $PreferenceInnovation$  (long_name='Preference Shock');

parameters alppha   ${\alpha}$      (long_name='capital share')
           betta    ${\beta}$       (long_name='discount factor')
           rho_a    ${\rho_a}$      (long_name='autocorrelation technology shock')
           rho_u    ${\rho_{u}}$    (long_name='autocorrelation cost push shock')
           rho_z    ${\rho_{z}}$    (long_name='autocorrelation preference shock')
           siggma   ${\sigma}$      (long_name='inverse EIS')
           varphi   ${\varphi}$     (long_name='inverse Frisch elasticity')
           eta      ${\eta}$        (long_name='semi-elasticity of money demand')
           epsilon  ${\epsilon}$    (long_name='demand elasticity')
           theta    ${\theta}$      (long_name='Calvo parameter')
           Omega    ${\Omega}$      (long_name='composite parameter Phillips curve')
           lambda   ${\lambda}$     (long_name='composite parameter Phillips curve')
           kappa    ${\kappa}$      (long_name='slope of the Phillips curve')
           vartheta ${\vartheta}$   (long_name='weight of x in the loss function');

% Calibration: Gali (2015), ch. 3 values (p. 67-75), rho_u as in ch. 5's persistent case
@#if !defined(siggma)
    @#define siggma = 1
@#endif
@#if !defined(varphi)
    @#define varphi = 5
@#endif
@#if !defined(theta)
    @#define theta = 0.75
@#endif
@#if !defined(rho_u)
    @#define rho_u = 0.5
@#endif
@#if !defined(rho_z)
    @#define rho_z = 0.5
@#endif
@#if !defined(rho_a)
    @#define rho_a = 0.9
@#endif
@#if !defined(betta)
    @#define betta = 0.99
@#endif
@#if !defined(eta)
    @#define eta = 3.77
@#endif
@#if !defined(alppha)
    @#define alppha = 0.25
@#endif
@#if !defined(epsilon)
    @#define epsilon = 9
@#endif
% shock scales in percent
@#if !defined(cost_push_shock_stderr)
    @#define cost_push_shock_stderr = 0.2
@#endif
@#if !defined(technology_shock_stderr)
    @#define technology_shock_stderr = 0.7
@#endif
@#if !defined(preference_shock_stderr)
    @#define preference_shock_stderr = 0.5
@#endif

siggma = @{siggma};
varphi = @{varphi};
theta = @{theta};
rho_u = @{rho_u};
rho_z = @{rho_z};
rho_a = @{rho_a};
betta = @{betta};
eta = @{eta};
alppha = @{alppha};
epsilon = @{epsilon};

model(linear);
#psi_n_ya = (1 + varphi) / (siggma * (1 - alppha) + varphi + alppha);
[name='Definition efficient interest rate, below equation (7)']
EfficientInterestRate = siggma * (EfficientOutput(+1) - EfficientOutput) + (1 - rho_z) * PreferenceShock;
[name='Definition efficient output']
EfficientOutput = psi_n_ya * TechnologyShock;
[name='Definition linking the output gaps']
OutputGap = WelfareRelevantOutputGap + (EfficientOutput - NaturalOutput);
[name='New Keynesian Phillips Curve eq. (2)']
PriceInflation = betta * PriceInflation(+1) + kappa * WelfareRelevantOutputGap + CostPushShock;
[name='Dynamic IS Curve eq. (7)']
WelfareRelevantOutputGap = WelfareRelevantOutputGap(+1) - 1 / siggma * (NominalInterestRate - PriceInflation(+1) - EfficientInterestRate);
[name='Definition real interest rate']
RealInterestRate = NominalInterestRate - PriceInflation(+1);
[name='Natural output implied by the cost-push shock']
CostPushShock = kappa * (EfficientOutput - NaturalOutput);
[name='Definition output gap']
OutputGap = OutputDeviationFromSteadyState - NaturalOutput;
[name='Cost push shock, equation (3)']
CostPushShock = rho_u * CostPushShock(-1) + CostPushInnovation;
[name='TFP shock']
TechnologyShock = rho_a * TechnologyShock(-1) + TechnologyInnovation;
[name='Production function']
OutputDeviationFromSteadyState = TechnologyShock + (1 - alppha) * HoursWorked;
[name='Money growth']
MoneyGrowthAnnual = 4 * (OutputDeviationFromSteadyState - OutputDeviationFromSteadyState(-1) - eta * (NominalInterestRate - NominalInterestRate(-1)) + PriceInflation);
[name='Annualized nominal interest rate']
AnnualNominalRate = 4 * NominalInterestRate;
[name='Annualized real interest rate']
AnnualRealRate = 4 * RealInterestRate;
[name='Annualized inflation']
AnnualInflation = 4 * PriceInflation;
[name='Preference shock']
PreferenceShock = rho_z * PreferenceShock(-1) - PreferenceInnovation;
end;

steady_state_model;
Omega = (1 - alppha) / (1 - alppha + alppha * epsilon);
lambda = (1 - theta) * (1 - betta * theta) / theta * Omega;
kappa = lambda * (siggma + (varphi + alppha) / (1 - alppha));
vartheta = kappa / epsilon;
end;

shocks;
    var CostPushInnovation; stderr @{cost_push_shock_stderr};
    var TechnologyInnovation; stderr @{technology_shock_stderr};
    var PreferenceInnovation; stderr @{preference_shock_stderr};
end;

planner_objective PriceInflation^2 + vartheta * WelfareRelevantOutputGap^2;
discretionary_policy(instruments=(NominalInterestRate), planner_discount=betta, discretionary_tol=1e-12, periods=@{periods}, irf=0, nomoments, nograph, nocorr, noprint);


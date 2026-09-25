/*
 * Baseline New Keynesian model of Jordi Gali (2015): Monetary Policy, Inflation, and the Business
 * Cycle, 2nd ed., Chapter 3 (interest-rate rule). Adapted from Johannes Pfeifer's DSGE_mod
 * implementation (GPL-3.0). Variables are percent deviations from the zero-inflation steady state.
 */

@#ifndef periods
    @#define periods = 1000
@#endif

var PriceInflation       $PriceInflation$       (long_name='Price Inflation')
    OutputGap            $OutputGap$            (long_name='Output Gap')
    NaturalOutput        $NaturalOutput$        (long_name='Natural Output')
    OutputDeviationFromSteadyState      $OutputDeviationFromSteadyState$      (long_name='Output Deviation From Steady State')
    NaturalInterestRate  $NaturalInterestRate$  (long_name='Natural Interest Rate')
    RealInterestRate     $RealInterestRate$     (long_name='Real Interest Rate')
    NominalInterestRate  $NominalInterestRate$  (long_name='Nominal Interest Rate')
    HoursWorked          $HoursWorked$          (long_name='Hours Worked')
    RealMoneyStock       $RealMoneyStock$       (long_name='Real Money Stock')
    MoneyGrowthAnnual    $MoneyGrowthAnnual$    (long_name='Money Growth Annualized')
    MonetaryShock        $MonetaryShock$        (long_name='Monetary Policy Shock')
    TechnologyShock      $TechnologyShock$      (long_name='Technology Shock')
    PreferenceShock      $PreferenceShock$      (long_name='Preference Shock')
    AnnualRealRate       $AnnualRealRate$       (long_name='Annualized Real Interest Rate')
    AnnualNominalRate    $AnnualNominalRate$    (long_name='Annualized Nominal Interest Rate')
    AnnualNaturalRate    $AnnualNaturalRate$    (long_name='Annualized Natural Interest Rate')
    AnnualInflation      $AnnualInflation$      (long_name='Annualized Inflation Rate')
    RealWage             $RealWage$             (long_name='Real Wage')
    PriceMarkup          $PriceMarkup$          (long_name='Price Markup');

varexo TechnologyInnovation  $TechnologyInnovation$  (long_name='Technology Shock')
       MonetaryInnovation    $MonetaryInnovation$    (long_name='Monetary Policy Shock')
       PreferenceInnovation  $PreferenceInnovation$  (long_name='Preference Shock');

parameters alppha   ${\alpha}$      (long_name='capital share')
           betta    ${\beta}$       (long_name='discount factor')
           rho_a    ${\rho_a}$      (long_name='autocorrelation technology shock')
           rho_nu   ${\rho_{\nu}}$  (long_name='autocorrelation monetary policy shock')
           rho_z    ${\rho_{z}}$    (long_name='autocorrelation preference shock')
           siggma   ${\sigma}$      (long_name='inverse EIS')
           varphi   ${\varphi}$     (long_name='inverse Frisch elasticity')
           phi_pi   ${\phi_{\pi}}$  (long_name='inflation feedback Taylor Rule')
           phi_y    ${\phi_{y}}$    (long_name='output feedback Taylor Rule')
           eta      ${\eta}$        (long_name='semi-elasticity of money demand')
           epsilon  ${\epsilon}$    (long_name='demand elasticity')
           theta    ${\theta}$      (long_name='Calvo parameter');

% Calibration: Gali (2015), pp. 67-75 (quarterly)
@#if !defined(siggma)
    @#define siggma = 1
@#endif
@#if !defined(varphi)
    @#define varphi = 5
@#endif
@#if !defined(phi_pi)
    @#define phi_pi = 1.5
@#endif
@#if !defined(phi_y)
    @#define phi_y = 0.125
@#endif
@#if !defined(theta)
    @#define theta = 0.75
@#endif
@#if !defined(rho_nu)
    @#define rho_nu = 0.5
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
@#if !defined(technology_shock_stderr)
    @#define technology_shock_stderr = 0.7
@#endif
@#if !defined(monetary_shock_stderr)
    @#define monetary_shock_stderr = 0.25
@#endif
@#if !defined(preference_shock_stderr)
    @#define preference_shock_stderr = 0.5
@#endif

siggma = @{siggma};
varphi = @{varphi};
phi_pi = @{phi_pi};
phi_y = @{phi_y};
theta = @{theta};
rho_nu = @{rho_nu};
rho_z = @{rho_z};
rho_a = @{rho_a};
betta = @{betta};
eta = @{eta};
alppha = @{alppha};
epsilon = @{epsilon};

model(linear);
#Omega = (1 - alppha) / (1 - alppha + alppha * epsilon);
#psi_n_ya = (1 + varphi) / (siggma * (1 - alppha) + varphi + alppha);
#lambda = (1 - theta) * (1 - betta * theta) / theta * Omega;
#kappa = lambda * (siggma + (varphi + alppha) / (1 - alppha));
[name='New Keynesian Phillips Curve eq. (22)']
PriceInflation = betta * PriceInflation(+1) + kappa * OutputGap;
[name='Dynamic IS Curve eq. (23)']
OutputGap = -1 / siggma * (NominalInterestRate - PriceInflation(+1) - NaturalInterestRate) + OutputGap(+1);
[name='Interest Rate Rule eq. (26)']
NominalInterestRate = phi_pi * PriceInflation + phi_y * OutputDeviationFromSteadyState + MonetaryShock;
[name='Definition natural rate of interest eq. (24)']
NaturalInterestRate = -siggma * psi_n_ya * (1 - rho_a) * TechnologyShock + (1 - rho_z) * PreferenceShock;
[name='Definition real interest rate']
RealInterestRate = NominalInterestRate - PriceInflation(+1);
[name='Definition natural output, eq. (20)']
NaturalOutput = psi_n_ya * TechnologyShock;
[name='Definition output gap']
OutputGap = OutputDeviationFromSteadyState - NaturalOutput;
[name='Monetary policy shock']
MonetaryShock = rho_nu * MonetaryShock(-1) + MonetaryInnovation;
[name='TFP shock']
TechnologyShock = rho_a * TechnologyShock(-1) + TechnologyInnovation;
[name='Production function (eq. 14)']
OutputDeviationFromSteadyState = TechnologyShock + (1 - alppha) * HoursWorked;
[name='Preference shock, p. 54']
PreferenceShock = rho_z * PreferenceShock(-1) - PreferenceInnovation;
[name='Real money demand (eq. 4)']
RealMoneyStock = OutputDeviationFromSteadyState - eta * NominalInterestRate;
[name='Money growth (derived from eq. (4))']
MoneyGrowthAnnual = 4 * (OutputDeviationFromSteadyState - OutputDeviationFromSteadyState(-1) - eta * (NominalInterestRate - NominalInterestRate(-1)) + PriceInflation);
[name='Annualized nominal interest rate']
AnnualNominalRate = 4 * NominalInterestRate;
[name='Annualized real interest rate']
AnnualRealRate = 4 * RealInterestRate;
[name='Annualized natural interest rate']
AnnualNaturalRate = 4 * NaturalInterestRate;
[name='Annualized inflation']
AnnualInflation = 4 * PriceInflation;
[name='FOC labor, eq. (2), with c = y (resource constraint eq. (12))']
RealWage = siggma * OutputDeviationFromSteadyState + varphi * HoursWorked;
[name='average price markup, eq. (18)']
PriceMarkup = -(siggma + (varphi + alppha) / (1 - alppha)) * OutputDeviationFromSteadyState + (1 + varphi) / (1 - alppha) * TechnologyShock;
end;

shocks;
    var TechnologyInnovation; stderr @{technology_shock_stderr};
    var MonetaryInnovation; stderr @{monetary_shock_stderr};
    var PreferenceInnovation; stderr @{preference_shock_stderr};
end;

steady;
check;

stoch_simul(order=1, periods=@{periods}, irf=0, nomoments, nofunctions, nograph, nocorr, noprint);

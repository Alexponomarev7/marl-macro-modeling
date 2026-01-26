% New Keynesian Model with Wage and Price Rigidities
% Based on Born & Pfeifer (2018): "The New Keynesian Wage Phillips Curve: Calvo vs. Rotemberg"
%
% Features:
%   - Calvo price and wage setting
%   - Taylor rule for monetary policy
%   - Technology, monetary, and preference shocks
%   - Linear approximation around zero-inflation steady state
%
% All variables are LOG-DEVIATIONS from steady state (linear model)
%
% Note: This version uses only EHL framework with Calvo wages for simplicity.
%       The original model has switches for SGU/EHL and Calvo/Rotemberg.

var PriceInflation       $PriceInflation$       (long_name='Price Inflation')
    OutputGap            $OutputGap$            (long_name='Output Gap')
    NaturalOutput        $NaturalOutput$        (long_name='Natural Output')
    Output               $Output$               (long_name='Output')
    OutputDeviation      $OutputDeviation$      (long_name='Output Deviation From Steady State')
    NaturalInterestRate  $NaturalInterestRate$  (long_name='Natural Interest Rate')
    RealInterestRate     $RealInterestRate$     (long_name='Real Interest Rate')
    NominalInterestRate  $NominalInterestRate$  (long_name='Nominal Interest Rate')
    HoursWorked          $HoursWorked$          (long_name='Hours Worked')
    RealMoneyStock       $RealMoneyStock$       (long_name='Real Money Stock')
    MoneyGrowthAnnual    $MoneyGrowthAnnual$    (long_name='Money Growth Annualized')
    NominalMoneyStock    $NominalMoneyStock$    (long_name='Nominal Money Stock')
    MonetaryShock        $MonetaryShock$        (long_name='Monetary Policy Shock')
    TechnologyShock      $TechnologyShock$      (long_name='Technology Shock')
    AnnualRealRate       $AnnualRealRate$       (long_name='Annualized Real Interest Rate')
    AnnualNominalRate    $AnnualNominalRate$    (long_name='Annualized Nominal Interest Rate')
    AnnualNaturalRate    $AnnualNaturalRate$    (long_name='Annualized Natural Interest Rate')
    AnnualInflation      $AnnualInflation$      (long_name='Annualized Inflation Rate')
    PreferenceShock      $PreferenceShock$      (long_name='Preference Shock')
    PriceLevel           $PriceLevel$           (long_name='Price Level')
    NominalWage          $NominalWage$          (long_name='Nominal Wage')
    Consumption          $Consumption$          (long_name='Consumption')
    RealWage             $RealWage$             (long_name='Real Wage')
    RealWageGap          $RealWageGap$          (long_name='Real Wage Gap')
    WageInflation        $WageInflation$        (long_name='Wage Inflation')
    NaturalWage          $NaturalWage$          (long_name='Natural Real Wage')
    PriceMarkup          $PriceMarkup$          (long_name='Price Markup')
    AnnualWageInflation  $AnnualWageInflation$  (long_name='Annualized Wage Inflation');

varexo TechnologyInnovation  $TechnologyInnovation$  (long_name='Technology Shock')
       MonetaryInnovation    $MonetaryInnovation$    (long_name='Monetary Policy Shock')
       PreferenceInnovation  $PreferenceInnovation$  (long_name='Preference Shock');

parameters alpha beta
           rho_technology rho_monetary rho_preference
           inverse_eis inverse_frisch
           taylor_inflation taylor_output
           money_demand_elasticity
           goods_elasticity labor_elasticity
           calvo_prices calvo_wages
           wage_slope;

@#if !defined(alpha)
    @#define alpha = 0.25
@#endif

@#if !defined(beta)
    @#define beta = 0.99
@#endif

@#if !defined(inverse_eis)
    @#define inverse_eis = 1.0
@#endif

@#if !defined(inverse_frisch)
    @#define inverse_frisch = 5.0
@#endif

@#if !defined(taylor_inflation)
    @#define taylor_inflation = 1.5
@#endif

@#if !defined(taylor_output)
    @#define taylor_output = 0.125
@#endif

@#if !defined(calvo_prices)
    @#define calvo_prices = 0.75
@#endif

@#if !defined(calvo_wages)
    @#define calvo_wages = 0.75
@#endif

@#if !defined(rho_technology)
    @#define rho_technology = 0.9
@#endif

@#if !defined(rho_monetary)
    @#define rho_monetary = 0.5
@#endif

@#if !defined(rho_preference)
    @#define rho_preference = 0.5
@#endif

@#if !defined(goods_elasticity)
    @#define goods_elasticity = 9.0
@#endif

@#if !defined(labor_elasticity)
    @#define labor_elasticity = 4.5
@#endif

@#if !defined(money_demand_elasticity)
    @#define money_demand_elasticity = 3.77
@#endif

@#if !defined(technology_shock_stderr)
    @#define technology_shock_stderr = 1.0
@#endif

@#if !defined(monetary_shock_stderr)
    @#define monetary_shock_stderr = 0.25
@#endif

@#if !defined(preference_shock_stderr)
    @#define preference_shock_stderr = 1.0
@#endif

alpha = @{alpha};
beta = @{beta};
inverse_eis = @{inverse_eis};
inverse_frisch = @{inverse_frisch};
taylor_inflation = @{taylor_inflation};
taylor_output = @{taylor_output};
calvo_prices = @{calvo_prices};
calvo_wages = @{calvo_wages};
rho_technology = @{rho_technology};
rho_monetary = @{rho_monetary};
rho_preference = @{rho_preference};
goods_elasticity = @{goods_elasticity};
labor_elasticity = @{labor_elasticity};
money_demand_elasticity = @{money_demand_elasticity};

% Wage Phillips Curve slope (EHL framework with Calvo wages)
wage_slope = (1 - calvo_wages) * (1 - beta * calvo_wages) 
             / (calvo_wages * (1 + labor_elasticity * inverse_frisch));

model(linear);

% Composite parameters (defined locally for clarity)
#Omega = (1 - alpha) / (1 - alpha + alpha * goods_elasticity);
#psi_n_ya = (1 + inverse_frisch) / (inverse_eis * (1 - alpha) + inverse_frisch + alpha);
#psi_n_wa = (1 - alpha * psi_n_ya) / (1 - alpha);
#lambda_p = (1 - calvo_prices) * (1 - beta * calvo_prices) / calvo_prices * Omega;
#aleph_p = alpha * lambda_p / (1 - alpha);
#aleph_w = wage_slope * (inverse_eis + inverse_frisch / (1 - alpha));

[name='New Keynesian Phillips Curve (prices)']
PriceInflation = beta * PriceInflation(+1) + aleph_p * OutputGap + lambda_p * RealWageGap;

[name='New Keynesian Wage Phillips Curve']
WageInflation = beta * WageInflation(+1) + aleph_w * OutputGap - wage_slope * RealWageGap;

[name='Dynamic IS Curve']
OutputGap = -1 / inverse_eis * (NominalInterestRate - PriceInflation(+1) - NaturalInterestRate) + OutputGap(+1);

[name='Taylor Rule']
NominalInterestRate = taylor_inflation * PriceInflation + taylor_output * OutputDeviation + MonetaryShock;

[name='Natural Interest Rate']
NaturalInterestRate = -inverse_eis * psi_n_ya * (1 - rho_technology) * TechnologyShock 
                        + (1 - rho_preference) * PreferenceShock;

[name='Real Wage Gap Evolution']
RealWageGap = RealWageGap(-1) + WageInflation - PriceInflation - (NaturalWage - NaturalWage(-1));

[name='Natural Wage']
NaturalWage = psi_n_wa * TechnologyShock;

[name='Price Markup']
PriceMarkup = -alpha / (1 - alpha) * OutputGap - RealWageGap;

[name='Real Wage Gap Definition']
RealWageGap = RealWage - NaturalWage;

[name='Real Interest Rate (Fisher equation)']
RealInterestRate = NominalInterestRate - PriceInflation(+1);

[name='Natural Output']
NaturalOutput = psi_n_ya * TechnologyShock;

[name='Output Gap Definition']
OutputGap = Output - NaturalOutput;

[name='Monetary Policy Shock Process']
MonetaryShock = rho_monetary * MonetaryShock(-1) + MonetaryInnovation;

[name='Technology Shock Process']
TechnologyShock = rho_technology * TechnologyShock(-1) + TechnologyInnovation;

[name='Production Function (log-linearized)']
Output = TechnologyShock + (1 - alpha) * HoursWorked;

[name='Preference Shock Process']
PreferenceShock = rho_preference * PreferenceShock(-1) - PreferenceInnovation;

[name='Money Growth (annualized)']
MoneyGrowthAnnual = 4 * (Output - Output(-1) - money_demand_elasticity * (NominalInterestRate - NominalInterestRate(-1)) + PriceInflation);

[name='Money Demand']
RealMoneyStock = Output - money_demand_elasticity * NominalInterestRate;

[name='Annualized Nominal Rate']
AnnualNominalRate = 4 * NominalInterestRate;

[name='Annualized Real Rate']
AnnualRealRate = 4 * RealInterestRate;

[name='Annualized Natural Rate']
AnnualNaturalRate = 4 * NaturalInterestRate;

[name='Annualized Price Inflation']
AnnualInflation = 4 * PriceInflation;

[name='Annualized Wage Inflation']
AnnualWageInflation = 4 * WageInflation;

[name='Output Deviation from Steady State']
OutputDeviation = Output - steady_state(Output);

[name='Price Level Evolution']
PriceInflation = PriceLevel - PriceLevel(-1);

[name='Goods Market Clearing']
Output = Consumption;

[name='Real Wage Definition']
RealWage = NominalWage - PriceLevel;

[name='Nominal Money Stock']
NominalMoneyStock = RealMoneyStock + PriceLevel;

end;

shocks;
    var TechnologyInnovation;
    stderr @{technology_shock_stderr};
    
    var MonetaryInnovation;
    stderr @{monetary_shock_stderr};
    
    var PreferenceInnovation;
    stderr @{preference_shock_stderr};
end;

stoch_simul(order=1, periods=@{periods}, drop=0, irf=0, nomoments, nofunctions, nograph, nocorr, noprint);

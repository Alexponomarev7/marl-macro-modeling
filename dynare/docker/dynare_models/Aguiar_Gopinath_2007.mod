% Aguiar-Gopinath (2007) Model
% Small Open Economy with Trend Shocks
%
% Key features:
%   - Trend shocks to productivity and government spending
%   - Debt-elastic interest rate
%   - Capital adjustment costs
%   - GHH-style preferences (consumption and leisure non-separable)
%
% Timing convention:
%   - Capital and Debt are predetermined (state variables)

var Consumption $Consumption$                                    (long_name='consumption')
    Capital $Capital$                                            (long_name='capital')
    Output $Output$                                              (long_name='output')
    Debt $Debt$                                                  (long_name='debt')
    BondPrice $BondPrice$                                        (long_name='bond Price')
    GovSpending $GovSpending$                                    (long_name='government Spending')
    Labor $Labor$                                                (long_name='labor')
    Utility $Utility$                                            (long_name='utility')
    Productivity $Productivity$                                  (long_name='productivity')
    MUConsumption $MUConsumption$                                (long_name='marginal Utility of Consumption')
    MULabor $MULabor$                                            (long_name='marginal Utility of Labor')
    ConsumptionGDP $ConsumptionGdp$                              (long_name='consumption to GDP Ratio')
    InvestmentGDP $InvestmentGDP$                                (long_name='investment to GDP Ratio')
    NetExports $NetExports$                                      (long_name='net Exports')
    Investment $Investment$                                      (long_name='investment')
    LogOutput $LogOutput$                                        (long_name='log Output')
    LogConsumption $LogConsumption$                              (long_name='log Consumption')
    LogInvestment $LogInvestment$                                (long_name='log Investment')
    OutputGrowth $OutputGrowth$                                  (long_name='output Growth');

predetermined_variables Capital Debt;

varexo ShockProductivity   $ShockProductivity$ (long_name='Productivity Shock')
       ShockGovSpending    $ShockGovSpending$ (long_name='Government Spending Shock');

parameters growth_rate ${\gamma}$              (long_name='Growth Rate')
           sigma ${\sigma}$                    (long_name='Risk Aversion')
           debt_share ${\alpha_D}$             (long_name='Debt Share')
           capital_share ${\alpha_K}$          (long_name='Capital Share')
           beta ${\beta}$                      (long_name='Discount Rate')
           depreciation ${\delta}$             (long_name='Depreciation Rate')
           capital_adjustment_cost ${\phi_K}$  (long_name='Capital Adjustment Cost')
           labor_share ${\alpha_N}$            (long_name='Labor Share')
           elasticity_substitution ${\eta}$    (long_name='Elasticity of Substitution')
           steady_state_debt ${\bar{D}}$       (long_name='Steady State Debt')
           interest_rate ${r}$                 (long_name='Interest Rate')
           persistence_productivity ${\rho_A}$ (long_name='Persistence of Productivity')
           persistence_gov_spending ${\rho_G}$ (long_name='Persistence of Government Spending');

@#if !defined(beta)
  @#define beta = 0.98
@#endif

@#if !defined(sigma)
  @#define sigma = 2
@#endif

@#if !defined(persistence_gov_spending)
  @#define persistence_gov_spending = 0.01
@#endif

@#if !defined(persistence_productivity)
  @#define persistence_productivity = 0.95
@#endif

@#if !defined(capital_share)
  @#define capital_share = 0.68
@#endif

@#if !defined(labor_share)
  @#define labor_share = 0.36
@#endif

@#if !defined(depreciation)
  @#define depreciation = 0.05
@#endif

@#if !defined(capital_adjustment_cost)
  @#define capital_adjustment_cost = 4
@#endif

@#if !defined(elasticity_substitution)
  @#define elasticity_substitution = 0.001
@#endif

@#if !defined(debt_share)
  @#define debt_share = 0.1
@#endif

@#if !defined(shock_gov_spending_stderr)
  @#define shock_gov_spending_stderr = 0.0281
@#endif

@#if !defined(shock_productivity_stderr)
  @#define shock_productivity_stderr = 0.0048
@#endif

beta = @{beta};
sigma = @{sigma};
persistence_gov_spending = @{persistence_gov_spending};
persistence_productivity = @{persistence_productivity};
capital_share = @{capital_share};
labor_share = @{labor_share};
depreciation = @{depreciation};
capital_adjustment_cost = @{capital_adjustment_cost};
elasticity_substitution = @{elasticity_substitution};
debt_share = @{debt_share};

growth_rate = log(1.0066);
interest_rate = 1/beta - 1;
steady_state_debt = 0.1;

model;

[name='Production function']
Output = exp(Productivity) * Capital^(1 - capital_share) * (exp(GovSpending) * Labor)^capital_share;

[name='Productivity process']
Productivity = persistence_productivity * Productivity(-1) + ShockProductivity;

[name='Government spending process']
GovSpending = (1 - persistence_gov_spending) * growth_rate + persistence_gov_spending * GovSpending(-1) + ShockGovSpending;

[name='Utility function']
Utility = (Consumption^labor_share * (1 - Labor)^(1 - labor_share))^(1 - sigma) / (1 - sigma);

[name='Marginal utility of consumption']
MUConsumption = labor_share * Utility / Consumption * (1 - sigma);

[name='Marginal utility of labor']
MULabor = -(1 - labor_share) * Utility / (1 - Labor) * (1 - sigma);

[name='Resource constraint']
Consumption + exp(GovSpending) * Capital(+1) = Output + (1 - depreciation) * Capital 
    - capital_adjustment_cost / 2 * (exp(GovSpending) * Capital(+1) / Capital - exp(growth_rate))^2 * Capital 
    - Debt + BondPrice * exp(GovSpending) * Debt(+1);

[name='Bond price (debt-elastic interest rate)']
1 / BondPrice = 1 + interest_rate + elasticity_substitution * (exp(Debt(+1) - steady_state_debt) - 1);

[name='Euler equation for capital']
MUConsumption * (1 + capital_adjustment_cost * (exp(GovSpending) * Capital(+1) / Capital - exp(growth_rate))) * exp(GovSpending) = 
    beta * exp(GovSpending * (labor_share * (1 - sigma))) * MUConsumption(+1) * 
    (1 - depreciation + (1 - capital_share) * Output(+1) / Capital(+1) 
    - capital_adjustment_cost / 2 * (2 * (exp(GovSpending(+1)) * Capital(+2) / Capital(+1) - exp(growth_rate)) * (-1) * exp(GovSpending(+1)) * Capital(+2) / Capital(+1) 
    + (exp(GovSpending(+1)) * Capital(+2) / Capital(+1) - exp(growth_rate))^2));
    
[name='Labor supply FOC']
MULabor + MUConsumption * capital_share * Output / Labor = 0;

[name='Euler equation for bonds']
MUConsumption * exp(GovSpending) * BondPrice = beta * exp(GovSpending * (labor_share * (1 - sigma))) * MUConsumption(+1);

[name='Investment definition']
Investment = exp(GovSpending) * Capital(+1) - (1 - depreciation) * Capital 
    + capital_adjustment_cost / 2 * (exp(GovSpending) * Capital(+1) / Capital - exp(growth_rate))^2 * Capital;

[name='Consumption to GDP ratio']
ConsumptionGDP = Consumption / Output;

[name='Investment to GDP ratio']
InvestmentGDP = Investment / Output;

[name='Net exports']
NetExports = (Debt - exp(GovSpending) * BondPrice * Debt(+1)) / Output;

[name='Log output']
LogOutput = log(Output);

[name='Log consumption']
LogConsumption = log(Consumption);

[name='Log investment']
LogInvestment = log(Investment);

[name='Output growth']
OutputGrowth = log(Output) - log(Output(-1)) + GovSpending(-1);

end;

steady_state_model;
    BondPrice = beta * exp(growth_rate)^(labor_share * (1 - sigma) - 1);
    output_capital_ratio = ((1 / BondPrice) - (1 - depreciation)) / (1 - capital_share);
    ConsumptionGDP = 1 + (1 - exp(growth_rate) - depreciation) * (1 / output_capital_ratio) - (1 - exp(growth_rate) * BondPrice) * debt_share;
    Labor = (capital_share * labor_share) / ((1 - labor_share) * ConsumptionGDP + capital_share * labor_share);
    Capital = (((exp(growth_rate)^capital_share) * (Labor^capital_share)) / output_capital_ratio)^(1 / capital_share);
    Output = Capital^(1 - capital_share) * (Labor * exp(growth_rate))^capital_share;
    Consumption = ConsumptionGDP * Output;
    Investment = (exp(growth_rate) - 1 + depreciation) * Capital;
    steady_state_debt = debt_share * Output;
    NetExports = (Output - Consumption - Investment) / Output;
    interest_rate = 1 / BondPrice - 1;
    Debt = steady_state_debt;
    Productivity = 0;
    GovSpending = growth_rate;
    Utility = (Consumption^labor_share * (1 - Labor)^(1 - labor_share))^(1 - sigma) / (1 - sigma);
    MUConsumption = labor_share * Utility / Consumption * (1 - sigma);
    MULabor = -(1 - labor_share) * Utility / (1 - Labor) * (1 - sigma);
    InvestmentGDP = (exp(GovSpending) * Capital - (1 - depreciation) * Capital) / Output;
    LogOutput = log(Output);
    LogConsumption = log(Consumption);
    LogInvestment = log(Investment);
    OutputGrowth = growth_rate;
end;

shocks;
    var ShockGovSpending; stderr @{shock_gov_spending_stderr};
    var ShockProductivity; stderr @{shock_productivity_stderr};
end;

steady;
check;

stoch_simul(order=1, periods=@{periods}, drop=0, irf=0, nomoments, nofunctions, nograph, nocorr, noprint);

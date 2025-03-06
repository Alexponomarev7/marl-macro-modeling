@#ifndef periods
    @#define periods = 100
@#endif

@#ifndef growth_rate
    @#define growth_rate = log(1.0066)
@#endif
@#ifndef risk_aversion
    @#define risk_aversion = 2
@#endif
@#ifndef debt_share
    @#define debt_share = 0.1
@#endif
@#ifndef capital_share
    @#define capital_share = 0.68
@#endif
@#ifndef discount_rate
    @#define discount_rate = 1/1.02
@#endif
@#ifndef depreciation
    @#define depreciation = 0.05
@#endif
@#ifndef capital_adjustment_cost
    @#define capital_adjustment_cost = 4
@#endif
@#ifndef labor_share
    @#define labor_share = 0.36
@#endif
@#ifndef elasticity_substitution
    @#define elasticity_substitution = 0.001
@#endif
@#ifndef steady_state_debt
    @#define steady_state_debt = 0.1
@#endif
@#ifndef persistence_productivity
    @#define persistence_productivity = 0.95
@#endif
@#ifndef persistence_gov_spending
    @#define persistence_gov_spending = 0.01
@#endif

var consumption ${C}$                  (long_name='Consumption')
    capital ${K}$                      (long_name='Capital')
    output ${Y}$                       (long_name='Output')
    debt ${D}$                         (long_name='Debt')
    bond_price ${Q}$                   (long_name='Bond Price')
    gov_spending ${G}$                 (long_name='Government Spending')
    labor ${N}$                        (long_name='Labor')
    utility ${U}$                      (long_name='Utility')
    productivity ${A}$                 (long_name='Productivity')
    marginal_utility_consumption ${\frac{\partial U}{\partial C}}$ (long_name='Marginal Utility of Consumption')
    marginal_utility_labor ${\frac{\partial U}{\partial N}}$       (long_name='Marginal Utility of Labor')
    consumption_to_gdp ${\frac{C}{Y}}$ (long_name='Consumption to GDP Ratio')
    investment_to_gdp ${\frac{I}{Y}}$  (long_name='Investment to GDP Ratio')
    net_exports ${NX}$                 (long_name='Net Exports')
    investment ${I}$                   (long_name='Investment')
    log_output ${\log(Y)}$             (long_name='Log Output')
    log_consumption ${\log(C)}$        (long_name='Log Consumption')
    log_investment ${\log(I)}$         (long_name='Log Investment')
    output_growth ${\Delta Y}$         (long_name='Output Growth');

predetermined_variables capital debt;

varexo shock_productivity ${\varepsilon_A}$ (long_name='Productivity Shock')
       shock_gov_spending ${\varepsilon_G}$ (long_name='Government Spending Shock');

parameters growth_rate ${\gamma}$              (long_name='Growth Rate')
           risk_aversion ${\sigma}$            (long_name='Risk Aversion')
           debt_share ${\alpha_D}$             (long_name='Debt Share')
           capital_share ${\alpha_K}$          (long_name='Capital Share')
           discount_rate ${\beta}$             (long_name='Discount Rate')
           depreciation ${\delta}$             (long_name='Depreciation Rate')
           capital_adjustment_cost ${\phi_K}$  (long_name='Capital Adjustment Cost')
           labor_share ${\alpha_N}$            (long_name='Labor Share')
           elasticity_substitution ${\eta}$    (long_name='Elasticity of Substitution')
           steady_state_debt ${\bar{D}}$       (long_name='Steady State Debt')
           interest_rate ${r}$                 (long_name='Interest Rate')
           persistence_productivity ${\rho_A}$ (long_name='Persistence of Productivity')
           persistence_gov_spending ${\rho_G}$ (long_name='Persistence of Government Spending');

growth_rate = @{growth_rate};
risk_aversion = @{risk_aversion};
debt_share = @{debt_share};
capital_share = @{capital_share};
discount_rate = @{discount_rate};
depreciation = @{depreciation};
capital_adjustment_cost = @{capital_adjustment_cost};
labor_share = @{labor_share};
elasticity_substitution = @{elasticity_substitution};
steady_state_debt = @{steady_state_debt};
interest_rate = 1/discount_rate - 1;
persistence_productivity = @{persistence_productivity};
persistence_gov_spending = @{persistence_gov_spending};

model;
    output = exp(productivity) * capital^(1 - capital_share) * (exp(gov_spending) * labor)^capital_share;
    productivity = persistence_productivity * productivity(-1) + shock_productivity;
    gov_spending = (1 - persistence_gov_spending) * growth_rate + persistence_gov_spending * gov_spending(-1) + shock_gov_spending;
    utility = (consumption^labor_share * (1 - labor)^(1 - labor_share))^(1 - risk_aversion) / (1 - risk_aversion);
    marginal_utility_consumption = labor_share * utility / consumption * (1 - risk_aversion);
    marginal_utility_labor = -(1 - labor_share) * utility / (1 - labor) * (1 - risk_aversion);
    consumption + exp(gov_spending) * capital(+1) = output + (1 - depreciation) * capital - capital_adjustment_cost / 2 * (exp(gov_spending) * capital(+1) / capital - exp(growth_rate))^2 * capital - debt + bond_price * exp(gov_spending) * debt(+1);
    1 / bond_price = 1 + interest_rate + elasticity_substitution * (exp(debt(+1) - steady_state_debt) - 1);
    marginal_utility_consumption * (1 + capital_adjustment_cost * (exp(gov_spending) * capital(+1) / capital - exp(growth_rate))) * exp(gov_spending) = discount_rate * exp(gov_spending * (labor_share * (1 - risk_aversion))) * marginal_utility_consumption(+1) * (1 - depreciation + (1 - capital_share) * output(+1) / capital(+1) - capital_adjustment_cost / 2 * (2 * (exp(gov_spending(+1)) * capital(+2) / capital(+1) - exp(growth_rate)) * (-1) * exp(gov_spending(+1)) * capital(+2) / capital(+1) + (exp(gov_spending(+1)) * capital(+2) / capital(+1) - exp(growth_rate))^2));
    marginal_utility_labor + marginal_utility_consumption * capital_share * output / labor = 0;
    marginal_utility_consumption * exp(gov_spending) * bond_price = discount_rate * exp(gov_spending * (labor_share * (1 - risk_aversion))) * marginal_utility_consumption(+1);
    investment = exp(gov_spending) * capital(+1) - (1 - depreciation) * capital + capital_adjustment_cost / 2 * (exp(gov_spending) * capital(+1) / capital - exp(growth_rate))^2 * capital;
    consumption_to_gdp = consumption / output;
    investment_to_gdp = investment / output;
    net_exports = (debt - exp(gov_spending) * bond_price * debt(+1)) / output;
    log_output = log(output);
    log_consumption = log(consumption);
    log_investment = log(investment);
    output_growth = log(output) - log(output(-1)) + gov_spending(-1);
end;

steady_state_model;
    bond_price = discount_rate * exp(growth_rate)^(labor_share * (1 - risk_aversion) - 1);
    output_capital_ratio = ((1 / bond_price) - (1 - depreciation)) / (1 - capital_share);
    consumption_to_gdp = 1 + (1 - exp(growth_rate) - depreciation) * (1 / output_capital_ratio) - (1 - exp(growth_rate) * bond_price) * debt_share;
    labor = (capital_share * labor_share) / (consumption_to_gdp - labor_share * consumption_to_gdp + capital_share * labor_share);
    capital = (((exp(growth_rate)^capital_share) * (labor^capital_share)) / output_capital_ratio)^(1 / capital_share);
    output = capital^(1 - capital_share) * (labor * exp(growth_rate))^capital_share;
    consumption = consumption_to_gdp * output;
    investment = (exp(growth_rate) - 1 + depreciation) * capital;
    steady_state_debt = debt_share * output;
    net_exports = (output - consumption - investment) / output;
    interest_rate = 1 / bond_price - 1;
    debt = steady_state_debt;
    productivity = 0;
    gov_spending = growth_rate;
    utility = (consumption^labor_share * (1 - labor)^(1 - labor_share))^(1 - risk_aversion) / (1 - risk_aversion);
    marginal_utility_consumption = labor_share * utility / consumption * (1 - risk_aversion);
    marginal_utility_labor = -(1 - labor_share) * utility / (1 - labor) * (1 - risk_aversion);
    investment_to_gdp = (exp(gov_spending) * capital - (1 - depreciation) * capital) / output;
    log_output = log(output);
    log_consumption = log(consumption);
    log_investment = log(investment);
    output_growth = growth_rate;
end;

shocks;
    var shock_gov_spending; stderr 0.0281;
    var shock_productivity; stderr 0.0048;
end;

steady;
check;
stoch_simul(irf=0, order=1, periods=@{periods}, nomoments, nofunctions, nograph, nocorr, noprint);
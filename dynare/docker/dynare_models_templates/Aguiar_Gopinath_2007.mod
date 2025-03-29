var consumption                    (long_name='Consumption')
    capital                        (long_name='Capital')
    output                         (long_name='Output')
    debt                           (long_name='Debt')
    bond_price                     (long_name='Bond Price')
    gov_spending                   (long_name='Government Spending')
    labor                          (long_name='Labor')
    utility                        (long_name='Utility')
    productivity                   (long_name='Productivity')
    marginal_utility_consumption   (long_name='Marginal Utility of Consumption')
    marginal_utility_labor         (long_name='Marginal Utility of Labor')
    consumption_to_gdp             (long_name='Consumption to GDP Ratio')
    investment_to_gdp              (long_name='Investment to GDP Ratio')
    net_exports                    (long_name='Net Exports')
    investment                     (long_name='Investment')
    log_output                     (long_name='Log Output')
    log_consumption                (long_name='Log Consumption')
    log_investment                 (long_name='Log Investment')
    output_growth                  (long_name='Output Growth');

predetermined_variables capital debt;

varexo  shock_gov_spending   (long_name='Government Spending Shock');

parameters growth_rate               (long_name='Growth Rate')
           risk_aversion             (long_name='Risk Aversion')
           debt_share                (long_name='Debt Share')
           capital_share             (long_name='Capital Share')
           discount_rate             (long_name='Discount Rate')
           depreciation              (long_name='Depreciation Rate')
           capital_adjustment_cost   (long_name='Capital Adjustment Cost')
           labor_share               (long_name='Labor Share')
           elasticity_substitution   (long_name='Elasticity of Substitution')
           steady_state_debt         (long_name='Steady State Debt')
           interest_rate             (long_name='Interest Rate')
           persistence_productivity  (long_name='Persistence of Productivity')
           persistence_gov_spending  (long_name='Persistence of Government Spending');

growth_rate = {growth_rate};
risk_aversion = {risk_aversion};
debt_share = {debt_share};
capital_share = {capital_share};
discount_rate = {discount_rate};
depreciation = {depreciation};
capital_adjustment_cost = {capital_adjustment_cost};
labor_share = {labor_share};
elasticity_substitution = {elasticity_substitution};
steady_state_debt = {steady_state_debt};
interest_rate = 1/discount_rate - 1;
persistence_productivity = {persistence_productivity};
persistence_gov_spending = {persistence_gov_spending};

model;
    output = exp(productivity) * capital^(1 - capital_share) * (exp(gov_spending) * labor)^capital_share;
    productivity = persistence_productivity * productivity(-1);
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
    var shock_gov_spending; periods {shock_periods}; values {shock_values};
end;

steady;
check;

perfect_foresight_setup(periods={periods});
perfect_foresight_solver;
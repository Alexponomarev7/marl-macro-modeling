@#ifndef periods
    @#define periods = 100
@#endif

@#ifndef fixed_WPC_slope
    @#define fixed_WPC_slope = 0
@#endif

@#ifndef SGU_framework
    @#define SGU_framework = 0
@#endif

@#ifndef Calvo
    @#define Calvo = 1
@#endif

@#ifndef taxes
    @#define taxes = 0
@#endif

var price_inflation       ${\pi^p}$               (long_name='Price Inflation')
    output_gap            ${\tilde y}$            (long_name='Output Gap')
    natural_output        ${y^{nat}}$             (long_name='Natural Output')
    output                ${y}$                   (long_name='Output')
    output_deviation      ${\hat y}$              (long_name='Output Deviation From Steady State')
    natural_interest_rate ${r^{nat}}$             (long_name='Natural Interest Rate')
    real_interest_rate    ${r^r}$                 (long_name='Real Interest Rate')
    nominal_interest_rate ${i}$                   (long_name='Nominal Interest Rate')
    hours_worked          ${n}$                   (long_name='Hours Worked')
    real_money_stock      ${(m-p)}$               (long_name='Real Money Stock')
    money_growth_annual   ${\Delta m}$            (long_name='Money Growth Annualized')
    nominal_money_stock   ${m}$                   (long_name='Nominal Money Stock')
    monetary_shock        ${\nu}$                 (long_name='AR(1) Monetary Policy Shock Process')
    technology_shock      ${a}$                   (long_name='AR(1) Technology Shock Process')
    annual_real_rate      ${r^{r,ann}}$           (long_name='Annualized Real Interest Rate')
    annual_nominal_rate   ${i^{ann}}$             (long_name='Annualized Nominal Interest Rate')
    annual_natural_rate   ${r^{nat,ann}}$         (long_name='Annualized Natural Interest Rate')
    annual_inflation      ${\pi^{p,ann}}$         (long_name='Annualized Inflation Rate')
    preference_shock      ${z}$                   (long_name='AR(1) Preference Shock Process')
    price_level           ${p}$                   (long_name='Price Level')
    nominal_wage          ${w}$                   (long_name='Nominal Wage')
    consumption           ${c}$                   (long_name='Consumption')
    real_wage             $\omega$                (long_name='Real Wage')
    real_wage_gap         ${\tilde \omega}$       (long_name='Real Wage Gap')
    wage_inflation        ${\pi^w}$               (long_name='Wage Inflation')
    natural_wage          ${w^{nat}}$             (long_name='Natural Real Wage')
    price_markup          ${\mu^p}$               (long_name='Markup')
    annual_wage_inflation ${\pi^{w,ann}}$         (long_name='Annualized Wage Inflation Rate')
;

varexo  technology_innovation ${\varepsilon_a}$   (long_name='Technology Shock')
        monetary_innovation   ${\varepsilon_\nu}$ (long_name='Monetary Policy Shock')
        preference_innovation ${\varepsilon_z}$   (long_name='Preference Shock Innovation')
       ;

parameters capital_share       ${\alpha}$         (long_name='Capital Share')
    discount_factor            ${\beta}$          (long_name='Discount Factor')
    persistence_technology     ${\rho_a}$         (long_name='Autocorrelation Technology Shock')
    persistence_monetary       ${\rho_{\nu}}$     (long_name='Autocorrelation Monetary Policy Shock')
    persistence_preference     ${\rho_{z}}$       (long_name='Autocorrelation Demand Shock')
    inverse_eis                ${\sigma}$         (long_name='Inverse EIS')
    inverse_frisch             ${\varphi}$        (long_name='Inverse Frisch Elasticity')
    taylor_inflation           ${\phi_{\pi}}$     (long_name='Inflation Feedback Taylor Rule')
    taylor_output              ${\phi_{y}}$       (long_name='Output Feedback Taylor Rule')
    money_demand_elasticity    ${\eta}$           (long_name='Semi-Elasticity Of Money Demand')
    goods_demand_elasticity    ${\epsilon_p}$     (long_name='Demand Elasticity Goods')
    calvo_prices               ${\theta_p}$       (long_name='Calvo Parameter Prices')
    labor_demand_elasticity    ${\epsilon_w}$     (long_name='Demand Elasticity Labor Services')
    @#if Calvo
        calvo_wages            ${\theta_w}$       (long_name='Calvo Parameter Wages')
    @#else
        rotemberg_wages        ${\phi_w}$         (long_name='Rotemberg Parameter Wages')
    @#endif
    labor_subsidy              ${\tau}$           (long_name='Labor Subsidy')
    wage_phillips_slope        ${\lambda_w}$      (long_name='Slope Of The Wage PC')
    steady_state_tax           ${bar {\tau^n}}$   (long_name='Steady State Labor Tax')
    ;

% Parameter values
inverse_eis = 1;
inverse_frisch = 5;
taylor_inflation = 1.5;
taylor_output = 0.125;
calvo_prices = 3/4;
persistence_monetary = 0.5;
persistence_preference = 0.5;
persistence_technology = 0.9;
discount_factor = 0.99;
money_demand_elasticity = 3.77;
capital_share = 1/4;
goods_demand_elasticity = 9;
labor_subsidy = 0;

labor_demand_elasticity = 4.5;
@#if fixed_WPC_slope
    wage_phillips_slope = 0.03;
@#else
    calvo_wages = 3/4;
@#endif

@#if taxes
    steady_state_tax = 0.99;
@#else
    steady_state_tax = 0;
@#endif

% Model equations
model(linear);
    #Omega = (1 - capital_share) / (1 - capital_share + capital_share * goods_demand_elasticity);
    #psi_n_ya = (1 + inverse_frisch) / (inverse_eis * (1 - capital_share) + inverse_frisch + capital_share);
    #psi_n_wa = (1 - capital_share * psi_n_ya) / (1 - capital_share);
    #lambda_p = (1 - calvo_prices) * (1 - discount_factor * calvo_prices) / calvo_prices * Omega;
    #aleph_p = capital_share * lambda_p / (1 - capital_share);
    #aleph_w = wage_phillips_slope * (inverse_eis + inverse_frisch / (1 - capital_share));

    price_inflation = discount_factor * price_inflation(+1) + aleph_p * output_gap + lambda_p * real_wage_gap;
    wage_inflation = discount_factor * wage_inflation(+1) + aleph_w * output_gap - wage_phillips_slope * real_wage_gap;
    output_gap = -1 / inverse_eis * (nominal_interest_rate - price_inflation(+1) - natural_interest_rate) + output_gap(+1);
    nominal_interest_rate = taylor_inflation * price_inflation + taylor_output * output_deviation + monetary_shock;
    natural_interest_rate = -inverse_eis * psi_n_ya * (1 - persistence_technology) * technology_shock + (1 - persistence_preference) * preference_shock;
    real_wage_gap = real_wage_gap(-1) + wage_inflation - price_inflation - (natural_wage - natural_wage(-1));
    natural_wage = psi_n_wa * technology_shock;
    price_markup = -capital_share / (1 - capital_share) * output_gap - real_wage_gap;
    real_wage_gap = real_wage - natural_wage;
    real_interest_rate = nominal_interest_rate - price_inflation(+1);
    natural_output = psi_n_ya * technology_shock;
    output_gap = output - natural_output;
    monetary_shock = persistence_monetary * monetary_shock(-1) + monetary_innovation;
    technology_shock = persistence_technology * technology_shock(-1) + technology_innovation;
    output = technology_shock + (1 - capital_share) * hours_worked;
    preference_shock = persistence_preference * preference_shock(-1) - preference_innovation;
    money_growth_annual = 4 * (output - output(-1) - money_demand_elasticity * (nominal_interest_rate - nominal_interest_rate(-1)) + price_inflation);
    real_money_stock = output - money_demand_elasticity * nominal_interest_rate;
    annual_nominal_rate = 4 * nominal_interest_rate;
    annual_real_rate = 4 * real_interest_rate;
    annual_natural_rate = 4 * natural_interest_rate;
    annual_inflation = 4 * price_inflation;
    annual_wage_inflation = 4 * wage_inflation;
    output_deviation = output - steady_state(output);
    price_inflation = price_level - price_level(-1);
    output = consumption;
    real_wage = nominal_wage - price_level;
    nominal_money_stock = real_money_stock + price_level;
end;

% Shock variances
shocks;
    var monetary_innovation = 0.25^2;
    var technology_innovation = 1^2;
    var preference_innovation = 1^2;
end;

% Steady state
steady_state_model;
@#if fixed_WPC_slope
    @#if SGU_framework
        @#if Calvo
            calvo_wages = get_Calvo_theta(wage_phillips_slope, labor_demand_elasticity, discount_factor, inverse_frisch, 1);
        @#else
            rotemberg_wages = (labor_demand_elasticity - 1) * (1 - steady_state_tax) / wage_phillips_slope * (1 - capital_share) * (goods_demand_elasticity - 1) / goods_demand_elasticity / tau_s_p * tau_s_w;
        @#endif
    @#else
        @#if Calvo
            calvo_wages = get_Calvo_theta(wage_phillips_slope, labor_demand_elasticity, discount_factor, inverse_frisch, 0);
        @#else
            rotemberg_wages = (labor_demand_elasticity - 1) * (1 - steady_state_tax) / wage_phillips_slope * (1 - capital_share) * (goods_demand_elasticity - 1) / goods_demand_elasticity / tau_s_p * tau_s_w;
        @#endif
    @#endif
@#else
    @#if SGU_framework
        @#if Calvo
            wage_phillips_slope = (1 - calvo_wages) * (1 - discount_factor * calvo_wages) / calvo_wages;
        @#else
            rotemberg_wages = (labor_demand_elasticity - 1) / ((1 - calvo_wages) * (1 - discount_factor * calvo_wages)) * (1 - steady_state_tax) * calvo_wages * (1 - capital_share) * (goods_demand_elasticity - 1) / goods_demand_elasticity / tau_s_p * tau_s_w;
            wage_phillips_slope = (labor_demand_elasticity - 1) / rotemberg_wages * (1 - steady_state_tax) * (1 - capital_share) * (goods_demand_elasticity - 1) / goods_demand_elasticity / tau_s_p * tau_s_w;
        @#endif
    @#else
        @#if Calvo
            wage_phillips_slope = (1 - calvo_wages) * (1 - discount_factor * calvo_wages) / (calvo_wages * (1 + labor_demand_elasticity * inverse_frisch));
        @#else
            rotemberg_wages = (labor_demand_elasticity - 1) / ((1 - calvo_wages) * (1 - discount_factor * calvo_wages)) * (1 - steady_state_tax) * calvo_wages * (1 - capital_share) * (goods_demand_elasticity - 1) / goods_demand_elasticity / tau_s_p * tau_s_w * (1 + labor_demand_elasticity * inverse_frisch);
            wage_phillips_slope = (labor_demand_elasticity - 1) * (1 - steady_state_tax) / rotemberg_wages * (1 - capital_share) * (goods_demand_elasticity - 1) / goods_demand_elasticity / tau_s_p * tau_s_w;
        @#endif
    @#endif
@#endif
end;


stoch_simul(order=1, irf=0, periods=@{periods}, nomoments, nofunctions, nograph, nocorr, noprint);
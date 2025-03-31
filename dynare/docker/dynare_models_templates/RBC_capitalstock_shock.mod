var y  (long_name='Output')
    c  (long_name='Consumption')
    k  (long_name='Capital')
    l  (long_name='Labor')
    z  (long_name='Total Factor Productivity')
    invest (long_name='Investment');

varexo eps_cap (long_name='Capital Shock');

parameters beta  (long_name='Discount Factor')
           psi   (long_name='Labor Disutility Parameter')
           delta (long_name='Depreciation Rate')
           alpha (long_name='Capital Share')
           rho   (long_name='Persistence TFP Shock')
           i_y   (long_name='Investment-Output Ratio')
           k_y   (long_name='Capital-Output Ratio')
           l_ss  (long_name='Steady State Labor')
           k_ss  (long_name='Steady State Capital')
           i_ss  (long_name='Steady State Investment')
           y_ss  (long_name='Steady State Output')
           c_ss  (long_name='Steady State Consumption');

alpha = {alpha};
i_y = {i_y};
k_y = {k_y};
rho = {rho};
beta = {beta};
delta = {delta};
l_ss = {l_ss};

model;
    psi * exp(c) / (1 - exp(l)) = (1 - alpha) * exp(z) * (exp(k) / exp(l))^alpha;
    1 / exp(c) = beta / exp(c(+1)) * (alpha * exp(z(+1)) * (exp(k(+1)) / exp(l(+1)))^(alpha - 1) + (1 - delta));
    exp(k) = exp(-eps_cap) * (exp(invest(-1)) + (1 - delta) * exp(k(-1)));
    exp(y) = exp(z) * exp(k)^alpha * exp(l)^(1 - alpha);
    z = rho * z(-1);
    exp(invest) = exp(y) - exp(c);
end;

steady_state_model;
    delta = i_y / k_y;
    beta = 1 / (alpha / k_y + (1 - delta));
    l_ss = 0.33;
    k_ss = ((1 / beta - (1 - delta)) / alpha)^(1 / (alpha - 1)) * l_ss;
    i_ss = delta * k_ss;
    y_ss = k_ss^alpha * l_ss^(1 - alpha);
    c_ss = k_ss^alpha * l_ss^(1 - alpha) - i_ss;
    psi = (1 - alpha) * (k_ss / l_ss)^alpha * (1 - l_ss) / c_ss;
    invest = log(i_ss);
    w = log((1 - alpha) * y_ss / l_ss);
    r = 4 * alpha * y_ss / k_ss;
    y = log(y_ss);
    k = log(k_ss);
    c = log(c_ss);
    l = log(l_ss);
    z = 0;
    ghat = 0;
end;

shocks;
    var eps_cap; periods {shock_periods}; values {shock_values};
end;

steady;
check;

perfect_foresight_setup(periods={periods});
perfect_foresight_solver;
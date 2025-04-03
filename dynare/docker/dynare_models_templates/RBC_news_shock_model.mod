var y  (long_name='Output')
    c  (long_name='Consumption')
    k  (long_name='Capital')
    l  (long_name='Labor')
    z  (long_name='Total Factor Productivity')
    r  (long_name='Annualized Interest Rate')
    w  (long_name='Real Wage')
    invest (long_name='Investment');

varexo eps_z_news (long_name='Anticipated TFP Shock');

parameters beta   (long_name='Discount Factor')
           psi    (long_name='Labor Disutility Parameter')
           sigma  (long_name='Risk Aversion')
           delta  (long_name='Depreciation Rate')
           alpha  (long_name='Capital Share')
           rhoz   (long_name='Persistence TFP Shock')
           gammax (long_name='Composite Growth Rate')
           n      (long_name='Population Growth')
           x      (long_name='Technology Growth')
           i_y    (long_name='Investment-Output Ratio')
           k_y    (long_name='Capital-Output Ratio');

sigma = {sigma};
alpha = {alpha};
i_y = {i_y};
k_y = {k_y};
x = {x};
n = {n};
rhoz = {rhoz};

model;
    exp(c)^(-sigma) = beta / gammax * exp(c(+1))^(-sigma) * (alpha * exp(z(+1)) * (exp(k) / exp(l(+1)))^(alpha - 1) + (1 - delta));
    psi * exp(c)^sigma * 1 / (1 - exp(l)) = exp(w);
    gammax * exp(k) = (1 - delta) * exp(k(-1)) + exp(invest);
    exp(y) = exp(invest) + exp(c);
    exp(y) = exp(z) * exp(k(-1))^alpha * exp(l)^(1 - alpha);
    exp(w) = (1 - alpha) * exp(y) / exp(l);
    r = 4 * alpha * exp(y) / exp(k(-1));
    z = rhoz * z(-1) + eps_z_news(-8);
end;

steady_state_model;
    gammax = (1 + n) * (1 + x);
    delta = i_y / k_y - x - n - n * x;
    beta = gammax / (alpha / k_y + (1 - delta));
    l_ss = 0.33;
    k_ss = ((1 / beta * gammax - (1 - delta)) / alpha)^(1 / (alpha - 1)) * l_ss;
    i_ss = (x + n + delta + n * x) * k_ss;
    y_ss = k_ss^alpha * l_ss^(1 - alpha);
    c_ss = y_ss - i_ss;
    psi = (1 - alpha) * (k_ss / l_ss)^alpha * (1 - l_ss) / c_ss^sigma;
    invest = log(i_ss);
    w = log((1 - alpha) * y_ss / l_ss);
    r = 4 * alpha * y_ss / k_ss;
    y = log(y_ss);
    k = log(k_ss);
    c = log(c_ss);
    l = log(l_ss);
    z = 0;
end;

shocks;
    var eps_z_news; periods {shock_periods}; values {shock_values};
end;

steady;
check;

perfect_foresight_setup(periods={periods});
perfect_foresight_solver;
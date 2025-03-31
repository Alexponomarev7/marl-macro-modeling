var y              (long_name='Output')
    c              (long_name='Consumption')
    k              (long_name='Capital')
    l              (long_name='Labor')
    z              (long_name='Total Factor Productivity')
    ghat           (long_name='Government Spending')
    r              (long_name='Annualized Interest Rate')
    w              (long_name='Real Wage')
    invest         (long_name='Investment')
    log_y          (long_name='Log Output')
    log_k          (long_name='Log Capital Stock')
    log_c          (long_name='Log Consumption')
    log_l          (long_name='Log Labor')
    log_w          (long_name='Log Real Wage')
    log_invest     (long_name='Log Investment');

varexo eps_g (long_name='Government Spending Shock');

parameters beta    (long_name='Discount Factor')
           psi     (long_name='Labor Disutility Parameter')
           sigma   (long_name='Risk Aversion')
           delta   (long_name='Depreciation Rate')
           alpha   (long_name='Capital Share')
           rhoz    (long_name='Persistence TFP Shock')
           rhog    (long_name='Persistence Government Spending Shock')
           gammax  (long_name='Composite Growth Rate')
           gshare  (long_name='Government Spending Share')
           n       (long_name='Population Growth')
           x       (long_name='Technology Growth')
           i_y     (long_name='Investment-Output Ratio')
           k_y     (long_name='Capital-Output Ratio')
           g_ss    (long_name='Steady State Government Spending');

sigma = {sigma};
alpha = {alpha};
i_y = {i_y};
k_y = {k_y};
x = {x};
n = {n};
rhoz = {rhoz};
rhog = {rhog};
gshare = {gshare};

model;
    c^(-sigma) = beta / gammax * c(+1)^(-sigma) * (alpha * exp(z(+1)) * (k / l(+1))^(alpha - 1) + (1 - delta));
    psi * c^sigma * 1 / (1 - l) = w;
    gammax * k = (1 - delta) * k(-1) + invest;
    y = invest + c + g_ss * exp(ghat);
    y = exp(z) * k(-1)^alpha * l^(1 - alpha);
    w = (1 - alpha) * y / l;
    r = 4 * alpha * y / k(-1);
    z = rhoz * z(-1);
    ghat = rhog * ghat(-1) + eps_g;
    log_y = log(y);
    log_k = log(k);
    log_c = log(c);
    log_l = log(l);
    log_w = log(w);
    log_invest = log(invest);
end;

steady_state_model;
    gammax = (1 + n) * (1 + x);
    delta = i_y / k_y - x - n - n * x;
    beta = (1 + x) * (1 + n) / (alpha / k_y + (1 - delta));
    l = 0.33;
    k = ((1 / beta * (1 + n) * (1 + x) - (1 - delta)) / alpha)^(1 / (alpha - 1)) * l;
    invest = (x + n + delta + n * x) * k;
    y = k^alpha * l^(1 - alpha);
    g = gshare * y;
    g_ss = g;
    c = (1 - gshare) * k^alpha * l^(1 - alpha) - invest;
    psi = (1 - alpha) * (k / l)^alpha * (1 - l) / c^sigma;
    w = (1 - alpha) * y / l;
    r = 4 * alpha * y / k;
    log_y = log(y);
    log_k = log(k);
    log_c = log(c);
    log_l = log(l);
    log_w = log(w);
    log_invest = log(invest);
    z = 0;
    ghat = 0;
end;

shocks;
    var eps_g; periods {shock_periods}; values {shock_values};
end;

steady;
check;

perfect_foresight_setup(periods={periods});
perfect_foresight_solver;
var w                  (long_name='Real Wage')
    r                  (long_name='Real Return On Capital')
    c                  (long_name='Consumption')
    k                  (long_name='Capital')
    h                  (long_name='Labor')
    m                  (long_name='Money Stock')
    p                  (long_name='Price Level')
    g                  (long_name='Growth Rate Of Money Stock')
    lambda             (long_name='Total Factor Productivity')
    y                  (long_name='Output');

varexo eps_lambda   (long_name='TFP Shock');

parameters beta        (long_name='Discount Factor')
           delta       (long_name='Depreciation Rate')
           theta       (long_name='Capital Share Production')
           A           (long_name='Labor Disutility Parameter')
           h_0         (long_name='Steady State Labor')
           B           (long_name='Composite Labor Disutility Parameter')
           gamma       (long_name='Autocorrelation TFP')
           pi          (long_name='Autocorrelation Money Growth')
           g_bar       (long_name='Steady State Growth Rate Of Money')
           D           (long_name='Coefficient Log Balances');

predetermined_variables k;

beta = {beta};
delta = {delta};
theta = {theta};
A = {A};
h_0 = {h_0};
gamma = {gamma};
pi = {pi};
g_bar = {g_bar};
D = {D};

model;
    c + k(+1) + m / p = w * h + r * k + (1 - delta) * k + m(-1) / p + (g - 1) * m(-1) / p;
    1 / c = beta * p / (c(+1) * p(+1)) + D * p / m;
    1 / c = -B / w;
    1 / c = beta / c(+1) * (r(+1) + 1 - delta);
    m = g * m(-1);
    y = lambda * k^theta * h^(1 - theta);
    w = (1 - theta) * lambda * k^theta * h^(-theta);
    r = theta * lambda * (k / h)^(theta - 1);
    log(g) = (1 - pi) * log(g_bar) + pi * log(g(-1));
    log(lambda) = gamma * log(lambda(-1)) + eps_lambda;
end;

steady_state_model;
    B = A * log(1 - h_0) / h_0;
    r = 1 / beta - 1 + delta;
    w = (1 - theta) * (r / theta)^(theta / (theta - 1));
    c = -w / B;
    mp = D * g_bar * c / (g_bar - beta);
    k = c / ((r * (1 - theta) / (w * theta))^(1 - theta) - delta);
    h = r * (1 - theta) / (w * theta) * k;
    y = k * (r * (1 - theta) / (w * theta))^(1 - theta);
    g = 1;
    lambda = 1;
    p = 1;
    m = p * D * g * c / (g - beta);
end;

shocks;
    var eps_lambda; periods {shock_periods}; values {shock_values};
end;

steady;
check;

perfect_foresight_setup(periods={periods});
perfect_foresight_solver;
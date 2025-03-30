var c                  (long_name='Consumption')
    w                  (long_name='Real Wage')
    r                  (long_name='Real Interest Rate')
    y                  (long_name='Output')
    h                  (long_name='Hours Worked')
    k                  (long_name='Capital Stock')
    invest             (long_name='Investment')
    lambda             (long_name='Total Factor Productivity')
    productivity       (long_name='Productivity');

varexo eps_a (long_name='TFP Shock');

parameters beta     (long_name='Discount Factor')
           delta    (long_name='Depreciation Rate')
           theta    (long_name='Capital Share')
           gammma    (long_name='AR Coefficient TFP')
           A            (long_name='Labor Disutility Parameter')
           h_0      (long_name='Full Time Workers In Steady State')
           sigma_eps  (long_name='TFP Shock Volatility')
           B           (long_name='Composite Labor Disutility Parameter');

beta = {beta};
delta = {delta};
theta = {theta};
gammma = {gammma};
A = {A};
sigma_eps = {sigma_eps};
h_0 = {h_0};

model;
    1 / c = beta * ((1 / c(+1)) * (r(+1) + (1 - delta)));
    (1 - theta) * (y / h) = B * c;
    c = y + (1 - delta) * k(-1) - k;
    k = (1 - delta) * k(-1) + invest;
    y = lambda * k(-1)^theta * h^(1 - theta);
    r = theta * (y / k(-1));
    w = (1 - theta) * (y / h);
    log(lambda) = gammma * log(lambda(-1)) + eps_a;
    productivity = y / h;
end;

steady_state_model;
    B = -A * (log(1 - h_0)) / h_0;
    lambda = 1;
    h = (1 - theta) * (1 / beta - (1 - delta)) / (B * (1 / beta - (1 - delta) - theta * delta));
    k = h * ((1 / beta - (1 - delta)) / (theta * lambda))^(1 / (theta - 1));
    invest = delta * k;
    y = lambda * k^theta * h^(1 - theta);
    c = y - delta * k;
    r = 1 / beta - (1 - delta);
    w = (1 - theta) * (y / h);
    productivity = y / h;
end;

shocks;
    var eps_a; periods {shock_periods}; values {shock_values};
end;

steady;
check;

perfect_foresight_setup(periods={periods});
perfect_foresight_solver;
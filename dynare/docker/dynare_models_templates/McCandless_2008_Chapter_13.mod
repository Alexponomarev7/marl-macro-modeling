var w             (long_name='Real Wage')
    r             (long_name='Real Return On Capital')
    c             (long_name='Real Consumption')
    k             (long_name='Capital Stock')
    h             (long_name='Hours Worked')
    m             (long_name='Money Stock')
    p             (long_name='Price Level')
    pstar         (long_name='Foreign Price Level')
    g             (long_name='Growth Rate Of Money Stock')
    lambda        (long_name='Total Factor Productivity')
    b             (long_name='Foreign Bonds')
    rf            (long_name='Foreign Interest Rate')
    e             (long_name='Exchange Rate')
    x             (long_name='Net Exports');

varexo eps_lambda        (long_name='TFP Shock');

parameters beta            (long_name='Discount Factor')
           delta           (long_name='Depreciation Rate')
           theta           (long_name='Capital Share Production')
           kappa           (long_name='Capital Adjustment Cost')
           a               (long_name='Risk Premium')
           B               (long_name='Composite Labor Disutility Parameter')
           gamma_lambda    (long_name='Autocorrelation TFP')
           gamma_g         (long_name='Autocorrelation Money Growth')
           gamma_pstar     (long_name='Autocorrelation Foreign Price')
           pistar          (long_name='Foreign Inflation')
           rstar           (long_name='Foreign Interest Rate')
           sigma_lambda    (long_name='Standard Deviation TFP Shock')
           sigma_g         (long_name='Standard Deviation Money Shock')
           sigma_pstar     (long_name='Standard Deviation Foreign Price Shock');

kappa = {kappa};
beta = {beta};
delta = {delta};
theta = {theta};
rstar = {rstar};
a = {a};
B = {B};
gamma_lambda = {gamma_lambda};
gamma_g = {gamma_g};
gamma_pstar = {gamma_pstar};
sigma_lambda = {sigma_lambda};
sigma_g = {sigma_g};
sigma_pstar = {sigma_pstar};

model;
    0 = e / (p(+1) * c(+1)) - beta * e(+1) * (1 + rf) / (p(+2) * c(+2));
    0 = p / (p(+1) * c(+1)) * (1 + kappa * (k - k(-1))) -
        beta * p(+1) / (p(+2) * c(+2)) * (r(+1) + (1 - delta) + kappa * (k(+1) - k));
    0 = B / w + beta * p / (p(+1) * c(+1));
    0 = p * c - m;
    0 = m / p + e * b / p + k + kappa / 2 * (k - k(-1))^2 -
        w * h - r * k(-1) - (1 - delta) * k(-1) - e * (1 + rf(-1)) * b(-1) / p;
    0 = w - (1 - theta) * lambda * k(-1)^theta * h^(-theta);
    0 = r - theta * lambda * k(-1)^(theta - 1) * h^(1 - theta);
    0 = b - (1 + rf(-1)) * b(-1) - pstar * x;
    0 = rf - rstar + a * b / pstar;
    0 = e - p / pstar;
    0 = m - g * m(-1);
    lambda = 1 - gamma_lambda + gamma_lambda * lambda(-1) + sigma_lambda * eps_lambda;
    g = (1 - gamma_g) * 1 + gamma_g * g(-1) + sigma_g;
    pstar = (1 - gamma_pstar) * 1 + gamma_pstar * pstar(-1) + sigma_pstar;
end;

steady_state_model;
    pistar = 1;
    r = 1 / beta - (1 - delta);
    rf = 1 / beta - 1;
    b = (rstar + 1 - 1 / beta) / a;
    x = ((1 - beta)^2 - (1 - beta) * beta * rstar) / (a * beta^2);
    w = (1 - theta) * (theta / r)^(theta / (1 - theta));
    c = beta * w / (-B * pistar);
    m_pss = c;
    k = theta * (m_pss - rf * b) / (r - theta * delta);
    h = r * (1 - theta) / (w * theta) * k;
    lambda = 1;
    g = 1;
    pstar = 1;
    p = 1;
    m = m_pss * p;
    e = 1;
end;

shocks;
    var eps_lambda; periods {shock_periods}; values {shock_values};
end;

steady;
check;

perfect_foresight_setup(periods={periods});
perfect_foresight_solver;
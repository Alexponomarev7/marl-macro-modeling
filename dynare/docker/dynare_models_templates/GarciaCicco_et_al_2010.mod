var c                (long_name='Consumption')
    k                (long_name='Capital')
    a                (long_name='Total Factor Productivity')
    h                (long_name='Labor')
    d                (long_name='Debt')
    y                (long_name='Output')
    invest           (long_name='Investment')
    tb               (long_name='Trade Balance')
    mu_c             (long_name='Marginal Utility of Consumption')
    tb_y             (long_name='Trade Balance to Output Ratio')
    g_y              (long_name='Output Growth Rate')
    g_c              (long_name='Consumption Growth Rate')
    g_invest         (long_name='Investment Growth Rate')
    g                (long_name='Technology Growth Rate')
    r                (long_name='Interest Rate')
    mu               (long_name='Country Premium Shock')
    nu               (long_name='Preference Shock')
;

predetermined_variables k d;

% Define parameters
parameters beta       (long_name='Discount Factor')
        gammma         (long_name='Intertemporal Elasticity of Substitution')
        delta         (long_name='Depreciation Rate')
        alpha         (long_name='Capital Elasticity of Production')
        psi           (long_name='Debt Elasticity of Interest Rate')
        omega         (long_name='Labor Disutility Parameter')
        theta         (long_name='Labor Utility Parameter')
        phi           (long_name='Capital Adjustment Cost Parameter')
        dbar          (long_name='Steady State Debt')
        gbar          (long_name='Steady State Technology Growth Rate')
        rho_a         (long_name='Persistence of Temporary Technology Shock')
        rho_g         (long_name='Persistence of Permanent Technology Shock')
        rho_nu        (long_name='Persistence of Preference Shock')
        rho_mu        (long_name='Persistence of Country Premium Shock')
        rho_s         (long_name='Persistence of Exogenous Spending Shock')
;

varexo eps_g  (long_name='Permanent Technology Shock')
;

beta = {beta};
gammma = {gammma};
delta = {delta};
alpha = {alpha};
omega = {omega};
theta = {theta};
phi = {phi};
dbar = {dbar};
gbar = {gbar};
rho_a = {rho_a};
rho_g = {rho_g};
rho_nu = {rho_nu};
rho_s = {rho_s};
rho_mu = {rho_mu};
psi = {psi};
s_share = {s_share};


model;
#RSTAR = 1/beta * gbar^gammma; % World interest rate

% 1. Interest Rate
r = RSTAR + psi*(exp(d-dbar) - 1) + exp(mu-1) - 1;

% 2. Marginal utility of consumption
mu_c = nu * (c - theta/omega*h^omega)^(-gammma);

% 3. Resource constraint
y = log(tb) + c + invest + phi/2 * (k(+1)/k*g - gbar)^2*k;

% 4. Trade balance
log(tb) = d - d(+1)*g/r;

% 5. Definition output
y = a*k^alpha*(g*h)^(1-alpha);

% 6. Definition investment
invest = k(+1)*g - (1-delta)*k;

% 7. Euler equation
mu_c = beta/g^gammma * r * mu_c(+1);

% 8. First order condition labor
theta*h^(omega-1) = (1-alpha)*a*g^(1-alpha)*(k/h)^alpha;

% 9. First order condition investment
mu_c*(1 + phi*(k(+1)/k*g - gbar)) = beta/g^gammma * mu_c(+1) * (1 - delta + alpha*a(+1)*(g(+1)*h(+1)/k(+1))^(1-alpha) + phi*k(+2)/k(+1)*g(+1)*(k(+2)/k(+1)*g(+1) - gbar) - phi/2*(k(+2)/k(+1)*g(+1) - gbar)^2);

% 10. Definition trade-balance to output ratio
log(tb_y) = log(tb)/y;

% 11. Output growth
g_y = y/y(-1)*g(-1);

% 12. Consumption growth
g_c = c/c(-1)*g(-1);

% 13. Investment growth
g_invest = invest/invest(-1)*g(-1);

% 14. LOM temporary TFP
log(a) = rho_a * log(a(-1));

% 15. LOM permanent TFP Growth
log(g/gbar) = rho_g * log(g(-1)/gbar) + eps_g;

% 16. Preference shock
log(nu) = rho_nu * log(nu(-1));

% 17. Exogenous stochastic country premium shock
log(mu) = rho_mu * log(mu(-1));

steady_state_model;
    r = 1/beta * gbar^gammma; % World interest rate
    d = dbar; % Foreign debt
    k_over_gh = ((gbar^gammma/beta - 1 + delta)/alpha)^(1/(alpha-1)); % k/(g*h)
    h = ((1-alpha)*gbar*k_over_gh^alpha/theta)^(1/(omega-1)); % Hours
    k = k_over_gh * gbar * h; % Capital
    invest = (gbar - 1 + delta) * k; % Investment
    y = k^alpha * (h * gbar)^(1-alpha); % Output
    s = 0;
    c = (gbar/r - 1) * d + y - s - invest; % Consumption
    tb = y - c - s - invest; % Trade balance
    tb_y = tb / y;
    mu_c = (c - theta/omega * h^omega)^(-gammma); % Marginal utility of wealth
    a = 1; % Productivity shock
    g = gbar; % Growth rate of nonstationary productivity shock
    g_c = g;
    g_invest = g;
    g_y = g;
    nu = 1;
    mu = 1;
    tb = exp(tb);
    tb_y = exp(tb_y);
end;

shocks;
    var eps_g; periods {shock_periods}; values {shock_values};
end;

steady;
check;

perfect_foresight_setup(periods={periods});
perfect_foresight_solver;
var lambda     (long_name='Lagrange Multiplier A')
    c          (long_name='Consumption')
    R          (long_name='Nominal Interest Rate')
    pi         (long_name='Inflation Rate')
    theta      (long_name='Market Tightness')
    v          (long_name='Vacancies')
    u          (long_name='Unemployment Rate')
    m          (long_name='Matches')
    q          (long_name='Meeting Rate Between Firms And Workers')
    n          (long_name='Employment')
    y_gross    (long_name='Gross Output A')
    y_net      (long_name='Gross Output B')
    mu         (long_name='Lagrange Multiplier B')
    z          (long_name='Log TFP')
    mc         (long_name='Marginal Costs')
    w          (long_name='Real Wage')
    g          (long_name='Government Spending')
    z_G        (long_name='Government Spending Shock')
    log_y_net  (long_name='Log Output')
    log_v      (long_name='Log Vacancies')
    log_w      (long_name='Log Wages')
    log_u      (long_name='Log Unemployment')
    log_theta  (long_name='Log Tightness A')
    log_pi     (long_name='Log Tightness B');

varexo epsilon_G    (long_name='Government Spending Shock');

parameters epsilon         (long_name='Substitution Elasticity')
           Psi             (long_name='Price Adjustment Costs')
           betta           (long_name='Discount Factor')
           xi              (long_name='Exponent Matching Function')
           varsigma        (long_name='Bargaining Power')
           rho             (long_name='Separation Rate')
           m_param         (long_name='Scaling Parameter In Matching Function')
           b_w_target      (long_name='Target Value Of Steady State B/W')
           b               (long_name='Real Unemployment Benefits')
           kappa           (long_name='Vacancy Posting Cost')
           lambda_par      (long_name='Wage Rigidity')
           g_share         (long_name='Steady State Government Spending Share')
           G_SS            (long_name='Steady State Government Spending')
           rho_G           (long_name='Persistence Government Spending')
           rho_Z           (long_name='Persistence TFP')
           siggma          (long_name='Risk Aversion')
           phi_R           (long_name='Interest Rate Smoothing')
           phi_pi          (long_name='Inflation Feedback')
           phi_y           (long_name='Output Feedback')
           phi_u           (long_name='Unemployment Feedback')
       
betta = {betta};
siggma = {siggma};
epsilon = {epsilon};
Psi = {Psi};
xi = {xi};
rho = {rho};
varsigma = {varsigma};
b_w_target = {b_w_target};
lambda_par = {lambda_par};
g_share = {g_share};
rho_Z = {rho_Z};
rho_G = {rho_G};
phi_R = {phi_R};
phi_pi = {phi_pi};
phi_y = {phi_y};
phi_u = {phi_u};

model;
    lambda = c^-siggma;
    1 / R = betta * (lambda(+1) / lambda) / pi(+1);
    theta = v / u;
    m = m_param * (u^xi) * (v^(1 - xi));
    q = m / v;
    y_gross = exp(z) * n;
    z = rho_Z * z(-1);
    n = (1 - rho) * (n(-1) + v(-1) * q(-1));
    u = 1 - n;
    mu = mc * exp(z) - w + betta * (lambda(+1) / lambda) * (1 - rho) * mu(+1);
    kappa / q = betta * (lambda(+1) / lambda) * (1 - rho) * mu(+1);
    1 - Psi * (pi - 1) * pi + betta * (lambda(+1) / lambda) * (Psi * (pi(+1) - 1) * pi(+1) * y_gross(+1) / y_gross) = (1 - mc) * epsilon;
    w = lambda_par * (varsigma * (mc * exp(z) + theta * kappa) + (1 - varsigma) * b) + (1 - lambda_par) * steady_state(w);
    y_net = c + g;
    y_net = y_gross - kappa * v - y_gross * (Psi / 2) * (pi - 1)^2;

    log(R / steady_state(R)) = phi_R * log(R(-1) / steady_state(R)) + (1 - phi_R) * (phi_pi * log(pi / 1) + phi_y * log(y_net / y_net(-1)) + phi_u * log(u / steady_state(u)));

    g = G_SS * exp(z_G);
    z_G = rho_G * z_G(-1) + epsilon_G;
    log_y_net = log(y_net);
    log_v = log(v);
    log_w = log(w);
    log_u = log(u);
    log_theta = log(theta);
    log_pi = log(pi);
end;
initval;
    R = 1 / betta;
end;

steady_state_model;
    R = 1 / betta;
    pi = R * betta;
    z = 0;
    z_G = 0;
    mc = (epsilon - 1 + Psi * (pi - 1) * pi * (1 - betta)) / epsilon;
    q = 0.7;
    theta = 0.6 / q;
    m_param = q * theta^xi;
    u = 1 / (1 + q * theta * (1 - rho) / rho);
    n = 1 - u;
    v = rho * n / ((1 - rho) * q);
    m = m_param * (u^xi) * (v^(1 - xi));
    w = ((1 - (1 - varsigma) * b_w_target) / (q * betta * (1 - rho) * varsigma * theta) + 1 / (1 - betta * (1 - rho)))^(-1) * mc * (varsigma / (q * betta * (1 - rho) * varsigma * theta) + 1 / (1 - betta * (1 - rho)));
    mu = (mc - w) / (1 - betta * (1 - rho));
    kappa = mu * (q * betta * (1 - rho));
    y_gross = n;
    y_net = y_gross - kappa * v;
    G_SS = g_share * y_net;
    g = G_SS;
    c = y_net - g;
    lambda = c^-siggma;
    b = b_w_target * w;
    log_y_net = log(y_net);
    log_v = log(v);
    log_w = log(w);
    log_u = log(u);
    log_theta = log(theta);
    log_pi = log(pi);
end;

ramsey_model(planner_discount = 0.99, instruments = (R));

shocks;
    var epsilon_G; periods {shock_periods}; values {shock_values};
end;

steady;
check;

perfect_foresight_setup(periods={periods});
perfect_foresight_solver;
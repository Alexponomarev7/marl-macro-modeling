@#ifndef periods
    @#define periods = 100
@#endif

@#ifndef Ramsey
    @#define Ramsey = 0
@#endif

var lambda  ${\Lambda}$     (long_name='Lagrange Multiplier A')
    c       ${c}$           (long_name='Consumption')
    R       ${R}$           (long_name='Nominal Interest Rate')
    pi      ${\pi}$         (long_name='Inflation Rate')
    theta   ${\theta}$      (long_name='Market Tightness')
    v       ${v}$           (long_name='Vacancies')
    u       ${u}$           (long_name='Unemployment Rate')
    m       ${m}$           (long_name='Matches')
    q       ${q}$           (long_name='Meeting Rate Between Firms And Workers')
    n       ${n}$           (long_name='Employment')
    y_gross ${y^{gross}}$   (long_name='Gross Output A')
    y_net   ${y^{net}}$     (long_name='Gross Output B')
    mu      ${\mu}$         (long_name='Lagrange Multiplier B')
    z       ${z}$           (long_name='Log TFP')
    mc      ${mc}$          (long_name='Marginal Costs')
    w       ${w}$           (long_name='Real Wage')
    g       ${g}$           (long_name='Government Spending')
    z_G     ${g}$           (long_name='Government Spending Shock')
    log_y_net   ${\log y}$  (long_name='Log Output')
    log_v       ${\log v}$  (long_name='Log Vacancies')
    log_w       ${\log w}$  (long_name='Log Wages')
    log_u       ${\log u}$  (long_name='Log Unemployment')
    log_theta   ${\log \theta}$ (long_name='Log Tightness A')
    log_pi      ${\log \pi}$    (long_name='Log Tightness B');

varexo epsilon_G    ${\varepsilon_G}$ (long_name='Government Spending Shock')
       epsilon_z    ${\varepsilon_Z}$ (long_name='Technology Shock');

parameters epsilon     ${\varepsilon}$ (long_name='Substitution Elasticity')
           Psi         ${\psi}$        (long_name='Price Adjustment Costs')
           betta       ${\beta}$       (long_name='Discount Factor')
           xi          ${\xi}$         (long_name='Exponent Matching Function')
           varsigma    ${\varsigma}$   (long_name='Bargaining Power')
           rho         ${\rho}$        (long_name='Separation Rate')
           m_param     ${m^{par}}$     (long_name='Scaling Parameter In Matching Function')
           b_w_target  ${\frac{b}{\bar w}}$ (long_name='Target Value Of Steady State B/W')
           b           $b$             (long_name='Real Unemployment Benefits')
           kappa       ${\kappa}$      (long_name='Vacancy Posting Cost')
           lambda_par  ${\lambda}$     (long_name='Wage Rigidity')
           g_share     ${\frac{\bar G}{\bar Y}}$ (long_name='Steady State Government Spending Share')
           G_SS        ${\bar G}$      (long_name='Steady State Government Spending')
           rho_G       ${\rho_G}$      (long_name='Persistence Government Spending')
           rho_Z       ${\rho_Z}$      (long_name='Persistence TFP')
           siggma      ${\sigma}$      (long_name='Risk Aversion')
           @#if Ramsey == 0
               phi_R   ${\phi_R}$      (long_name='Interest Rate Smoothing')
               phi_pi  ${\phi_\pi}$    (long_name='Inflation Feedback')
               phi_y   ${\phi_y}$      (long_name='Output Feedback')
               phi_u   ${\phi_u}$      (long_name='Unemployment Feedback')
           @#endif
        ;

betta = 0.99;
siggma = 2;
epsilon = 6;
Psi = 50;
xi = 0.4;
rho = 0.08;
varsigma = 0.5;
b_w_target = 0.5;
lambda_par = 0.6;
g_share = 0.25;
rho_Z = 0.95;
rho_G = 0.9;

@#if Ramsey == 0
    phi_R = 0.9;
    phi_pi = 5;
    phi_y = 0;
    phi_u = 0;
@#endif

model;
    lambda = c^-siggma;
    1 / R = betta * (lambda(+1) / lambda) / pi(+1);
    theta = v / u;
    m = m_param * (u^xi) * (v^(1 - xi));
    q = m / v;
    y_gross = exp(z) * n;
    z = rho_Z * z(-1) + epsilon_z;
    n = (1 - rho) * (n(-1) + v(-1) * q(-1));
    u = 1 - n;
    mu = mc * exp(z) - w + betta * (lambda(+1) / lambda) * (1 - rho) * mu(+1);
    kappa / q = betta * (lambda(+1) / lambda) * (1 - rho) * mu(+1);
    1 - Psi * (pi - 1) * pi + betta * (lambda(+1) / lambda) * (Psi * (pi(+1) - 1) * pi(+1) * y_gross(+1) / y_gross) = (1 - mc) * epsilon;
    w = lambda_par * (varsigma * (mc * exp(z) + theta * kappa) + (1 - varsigma) * b) + (1 - lambda_par) * steady_state(w);
    y_net = c + g;
    y_net = y_gross - kappa * v - y_gross * (Psi / 2) * (pi - 1)^2;
    @#if Ramsey == 0
        log(R / steady_state(R)) = phi_R * log(R(-1) / steady_state(R)) + (1 - phi_R) * (phi_pi * log(pi / 1) + phi_y * log(y_net / y_net(-1)) + phi_u * log(u / steady_state(u)));
    @#endif
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
    @#if Ramsey == 0
        R = 1 / betta;
    @#endif
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

@#if Ramsey
    ramsey_model(planner_discount = 0.99, instruments = (R));
@#endif

steady;
check;

shocks;
    var epsilon_z;
    stderr 0.008;
    var epsilon_G;
    stderr 0.008;
end;

@#if Ramsey
    planner_objective(log(c));
@#endif

shocks;
    var epsilon_z;
    stderr 1;
    var epsilon_G;
    stderr 1;
end;

stoch_simul(order=1, irf=60, periods=@{periods}, pruning, nomoments, nofunctions, nograph, nocorr, noprint);
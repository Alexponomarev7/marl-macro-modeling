@#ifndef periods
    @#define periods = 100
@#endif

@#ifndef model1
    @#define model1 = 0
@#endif

@#ifndef model1a
    @#define model1a = 0
@#endif

@#ifndef model2
    @#define model2 = 0
@#endif

@#ifndef model3
    @#define model3 = 0
@#endif

@#ifndef model4
    @#define model4 = 0
@#endif

@#ifndef model5
    @#define model5 = 0
@#endif


var c ${c}$ (long_name='Consumption')
    h ${h}$ (long_name='Hours Worked')
    y ${y}$ (long_name='Output')
    i ${i}$ (long_name='Investment')
    k ${k}$ (long_name='Capital')
    a ${a}$ (long_name='Total Factor Productivity')
    lambda ${\lambda}$ (long_name='Marginal Utility')
    util ${util}$ (long_name='Utility')
    @#if model1 == 1 || model1a == 1 || model2 == 1 || model3 == 1 || model5 == 1
        d ${d}$ (long_name='Debt')
        tb_y ${tb_y}$ (long_name='Trade Balance To Output Ratio')
        ca_y ${ca_y}$ (long_name='Current Account To Output Ratio')
        r ${r}$ (long_name='Interest Rate')
    @#endif
    @#if model1 == 1 || model1a == 1
        beta_fun ${\beta}$ (long_name='Discount Factor')
    @#endif
    @#if model1 == 1
        eta ${\eta}$ (long_name='Lagrange Multiplier On Discount Factor')
    @#endif
    @#if model2 == 1
        riskpremium ${riskpremium}$ (long_name='Risk Premium')
    @#endif
    @#if model4 == 1
        tb_y ${tb_y}$ (long_name='Trade Balance To Output Ratio')
    @#endif
    ;

varexo e ${\varepsilon}$ (long_name='TFP Shock');

parameters gamma ${\gamma}$ (long_name='Risk Aversion')
           omega ${\omega}$ (long_name='Frisch Elasticity Parameter')
           rho ${\rho}$ (long_name='Persistence TFP Shock')
           sigma_tfp ${\sigma_{a}}$ (long_name='Standard Deviation TFP Shock')
           delta ${\delta}$ (long_name='Depreciation Rate')
           psi_1 ${\psi_1}$ (long_name='Elasticity Discount Factor')
           psi_2 ${\psi_2}$ (long_name='Risk Premium Parameter')
           alpha ${\alpha}$ (long_name='Labor Share')
           phi ${\phi}$ (long_name='Capital Adjustment Cost Parameter')
           psi_3 ${\psi_3}$ (long_name='Portfolio Holding Cost Parameter')
           psi_4 ${\psi_4}$ (long_name='Complete Markets Parameter')
           r_bar ${\bar r}$ (long_name='Steady State Interest Rate')
           d_bar ${\bar d}$ (long_name='Steady State Debt')
           beta ${\beta}$ (long_name='Discount Factor');

gamma = 2;
omega = 1.455;
rho = 0.42;
sigma_tfp = 0.0129;
delta = 0.1;
alpha = 0.32;
phi = 0.028;
r_bar = 0.04;
d_bar = 0.7442;
psi_2 = 0.000742;
psi_3 = 0.00074;
psi_4 = 0;

@#if model1 == 1 || model1a == 1
    psi_1 = 0;
@#endif

@#if model1 == 1
model;
    d = (1 + exp(r(-1))) * d(-1) - exp(y) + exp(c) + exp(i) + (phi / 2) * (exp(k) - exp(k(-1)))^2;
    exp(y) = exp(a) * (exp(k(-1))^alpha) * (exp(h)^(1 - alpha));
    exp(k) = exp(i) + (1 - delta) * exp(k(-1));
    exp(lambda) = beta_fun * (1 + exp(r)) * exp(lambda(+1));
    exp(lambda) = (exp(c) - (exp(h)^omega) / omega)^(-gamma) - eta * (-psi_1 * (1 + exp(c) - omega^(-1) * exp(h)^omega)^(-psi_1 - 1));
    eta = -util(+1) + eta(+1) * beta_fun(+1);
    ((exp(c) - (exp(h)^omega) / omega)^(-gamma)) * (exp(h)^(omega - 1)) + eta * (-psi_1 * (1 + exp(c) - omega^(-1) * exp(h)^omega)^(-psi_1 - 1) * (-exp(h)^(omega - 1))) = exp(lambda) * (1 - alpha) * exp(y) / exp(h);
    exp(lambda) * (1 + phi * (exp(k) - exp(k(-1)))) = beta_fun * exp(lambda(+1)) * (alpha * exp(y(+1)) / exp(k) + 1 - delta + phi * (exp(k(+1)) - exp(k)));
    a = rho * a(-1) + sigma_tfp * e;
    beta_fun = (1 + exp(c) - omega^(-1) * exp(h)^omega)^(-psi_1);
    util = (((exp(c) - omega^(-1) * exp(h)^omega)^(1 - gamma)) - 1) / (1 - gamma);
    exp(r) = r_bar;
    tb_y = 1 - ((exp(c) + exp(i) + (phi / 2) * (exp(k) - exp(k(-1)))^2) / exp(y));
    ca_y = (1 / exp(y)) * (d(-1) - d);
end;

steady_state_model;
    r = log(r_bar);
    d = d_bar;
    h = log(((1 - alpha) * (alpha / (r_bar + delta))^(alpha / (1 - alpha)))^(1 / (omega - 1)));
    k = log(exp(h) / (((r_bar + delta) / alpha)^(1 / (1 - alpha))));
    y = log((exp(k)^alpha) * (exp(h)^(1 - alpha)));
    i = log(delta * exp(k));
    c = log(exp(y) - exp(i) - r_bar * d);
    tb_y = 1 - ((exp(c) + exp(i)) / exp(y));
    util = (((exp(c) - omega^(-1) * exp(h)^omega)^(1 - gamma)) - 1) / (1 - gamma);
    psi_1 = -log(1 / (1 + r_bar)) / (log((1 + exp(c) - omega^(-1) * exp(h)^omega)));
    beta_fun = (1 + exp(c) - omega^(-1) * exp(h)^omega)^(-psi_1);
    eta = -util / (1 - beta_fun);
    lambda = log((exp(c) - (exp(h)^omega) / omega)^(-gamma) - eta * (-psi_1 * (1 + exp(c) - omega^(-1) * exp(h)^omega)^(-psi_1 - 1)));
    a = 0;
    ca_y = 0;
end;
@#endif

@#if model1a == 1
model;
    d = (1 + exp(r(-1))) * d(-1) - exp(y) + exp(c) + exp(i) + (phi / 2) * (exp(k) - exp(k(-1)))^2;
    exp(y) = exp(a) * (exp(k(-1))^alpha) * (exp(h)^(1 - alpha));
    exp(k) = exp(i) + (1 - delta) * exp(k(-1));
    exp(lambda) = beta_fun * (1 + exp(r)) * exp(lambda(+1));
    exp(lambda) = (exp(c) - (exp(h)^omega) / omega)^(-gamma);
    ((exp(c) - (exp(h)^omega) / omega)^(-gamma)) * (exp(h)^(omega - 1)) = exp(lambda) * (1 - alpha) * exp(y) / exp(h);
    exp(lambda) * (1 + phi * (exp(k) - exp(k(-1)))) = beta_fun * exp(lambda(+1)) * (alpha * exp(y(+1)) / exp(k) + 1 - delta + phi * (exp(k(+1)) - exp(k)));
    a = rho * a(-1) + sigma_tfp * e;
    beta_fun = (1 + exp(c) - omega^(-1) * exp(h)^omega)^(-psi_1);
    util = (((exp(c) - omega^(-1) * exp(h)^omega)^(1 - gamma)) - 1) / (1 - gamma);
    exp(r) = r_bar;
    tb_y = 1 - ((exp(c) + exp(i) + (phi / 2) * (exp(k) - exp(k(-1)))^2) / exp(y));
    ca_y = (1 / exp(y)) * (d(-1) - d);
end;

steady_state_model;
    r = log(r_bar);
    d = d_bar;
    h = log(((1 - alpha) * (alpha / (r_bar + delta))^(alpha / (1 - alpha)))^(1 / (omega - 1)));
    k = log(exp(h) / (((r_bar + delta) / alpha)^(1 / (1 - alpha))));
    y = log((exp(k)^alpha) * (exp(h)^(1 - alpha)));
    i = log(delta * exp(k));
    c = log(exp(y) - exp(i) - r_bar * d);
    tb_y = 1 - ((exp(c) + exp(i)) / exp(y));
    util = (((exp(c) - omega^(-1) * exp(h)^omega)^(1 - gamma)) - 1) / (1 - gamma);
    psi_1 = -log(1 / (1 + r_bar)) / (log((1 + exp(c) - omega^(-1) * exp(h)^omega)));
    beta_fun = (1 + exp(c) - omega^(-1) * exp(h)^omega)^(-psi_1);
    lambda = log((exp(c) - (exp(h)^omega) / omega)^(-gamma));
    a = 0;
    ca_y = 0;
end;
@#endif

@#if model2 == 1
model;
    d = (1 + exp(r(-1))) * d(-1) - exp(y) + exp(c) + exp(i) + (phi / 2) * (exp(k) - exp(k(-1)))^2;
    exp(y) = exp(a) * (exp(k(-1))^alpha) * (exp(h)^(1 - alpha));
    exp(k) = exp(i) + (1 - delta) * exp(k(-1));
    exp(lambda) = beta * (1 + exp(r)) * exp(lambda(+1));
    (exp(c) - (exp(h)^omega) / omega)^(-gamma) = exp(lambda);
    ((exp(c) - (exp(h)^omega) / omega)^(-gamma)) * (exp(h)^(omega - 1)) = exp(lambda) * (1 - alpha) * exp(y) / exp(h);
    exp(lambda) * (1 + phi * (exp(k) - exp(k(-1)))) = beta * exp(lambda(+1)) * (alpha * exp(y(+1)) / exp(k) + 1 - delta + phi * (exp(k(+1)) - exp(k)));
    a = rho * a(-1) + sigma_tfp * e;
    exp(r) = r_bar + riskpremium;
    riskpremium = psi_2 * (exp(d - d_bar) - 1);
    tb_y = 1 - ((exp(c) + exp(i) + (phi / 2) * (exp(k) - exp(k(-1)))^2) / exp(y));
    ca_y = (1 / exp(y)) * (d(-1) - d);
    util = (((exp(c) - omega^(-1) * exp(h)^omega)^(1 - gamma)) - 1) / (1 - gamma);
end;

steady_state_model;
    beta = 1 / (1 + r_bar);
    r = log((1 - beta) / beta);
    d = d_bar;
    h = log(((1 - alpha) * (alpha / (r_bar + delta))^(alpha / (1 - alpha)))^(1 / (omega - 1)));
    k = log(exp(h) / (((r_bar + delta) / alpha)^(1 / (1 - alpha))));
    y = log((exp(k)^alpha) * (exp(h)^(1 - alpha)));
    i = log(delta * exp(k));
    c = log(exp(y) - exp(i) - r_bar * d);
    tb_y = 1 - ((exp(c) + exp(i)) / exp(y));
    util = (((exp(c) - omega^(-1) * exp(h)^omega)^(1 - gamma)) - 1) / (1 - gamma);
    lambda = log((exp(c) - (exp(h)^omega) / omega)^(-gamma));
    a = 0;
    ca_y = 0;
    riskpremium = 0;
end;
@#endif

@#if model3 == 1
model;
    d = (1 + exp(r(-1))) * d(-1) - exp(y) + exp(c) + exp(i) + (phi / 2) * (exp(k) - exp(k(-1)))^2 + psi_3 / 2 * (d - d_bar)^2;
    exp(y) = exp(a) * (exp(k(-1))^alpha) * (exp(h)^(1 - alpha));
    exp(k) = exp(i) + (1 - delta) * exp(k(-1));
    exp(lambda) * (1 - psi_3 * (d - d_bar)) = beta * (1 + exp(r)) * exp(lambda(+1));
    (exp(c) - (exp(h)^omega) / omega)^(-gamma) = exp(lambda);
    ((exp(c) - (exp(h)^omega) / omega)^(-gamma)) * (exp(h)^(omega - 1)) = exp(lambda) * (1 - alpha) * exp(y) / exp(h);
    exp(lambda) * (1 + phi * (exp(k) - exp(k(-1)))) = beta * exp(lambda(+1)) * (alpha * exp(y(+1)) / exp(k) + 1 - delta + phi * (exp(k(+1)) - exp(k)));
    a = rho * a(-1) + sigma_tfp * e;
    exp(r) = r_bar;
    tb_y = 1 - ((exp(c) + exp(i) + (phi / 2) * (exp(k) - exp(k(-1)))^2) / exp(y));
    ca_y = (1 / exp(y)) * (d(-1) - d);
    util = (((exp(c) - omega^(-1) * exp(h)^omega)^(1 - gamma)) - 1) / (1 - gamma);
end;

steady_state_model;
    beta = 1 / (1 + r_bar);
    r = log((1 - beta) / beta);
    d = d_bar;
    h = log(((1 - alpha) * (alpha / (r_bar + delta))^(alpha / (1 - alpha)))^(1 / (omega - 1)));
    k = log(exp(h) / (((r_bar + delta) / alpha)^(1 / (1 - alpha))));
    y = log((exp(k)^alpha) * (exp(h)^(1 - alpha)));
    i = log(delta * exp(k));
    c = log(exp(y) - exp(i) - r_bar * d);
    tb_y = 1 - ((exp(c) + exp(i)) / exp(y));
    util = (((exp(c) - omega^(-1) * exp(h)^omega)^(1 - gamma)) - 1) / (1 - gamma);
    lambda = log((exp(c) - (exp(h)^omega) / omega)^(-gamma));
    a = 0;
    ca_y = 0;
end;
@#endif

@#if model4 == 1
model;
    exp(y) = exp(a) * (exp(k(-1))^alpha) * (exp(h)^(1 - alpha));
    exp(k) = exp(i) + (1 - delta) * exp(k(-1));
    (exp(c) - (exp(h)^omega) / omega)^(-gamma) = exp(lambda);
    ((exp(c) - (exp(h)^omega) / omega)^(-gamma)) * (exp(h)^(omega - 1)) = exp(lambda) * (1 - alpha) * exp(y) / exp(h);
    exp(lambda) * (1 + phi * (exp(k) - exp(k(-1)))) = beta * exp(lambda(+1)) * (alpha * exp(y(+1)) / exp(k) + 1 - delta + phi * (exp(k(+1)) - exp(k)));
    exp(lambda) = psi_4;
    a = rho * a(-1) + sigma_tfp * e;
    tb_y = 1 - ((exp(c) + exp(i) + (phi / 2) * (exp(k) - exp(k(-1)))^2) / exp(y));
    util = (((exp(c) - omega^(-1) * exp(h)^omega)^(1 - gamma)) - 1) / (1 - gamma);
end;

steady_state_model;
    beta = 1 / (1 + r_bar);
    h = log(((1 - alpha) * (alpha / (r_bar + delta))^(alpha / (1 - alpha)))^(1 / (omega - 1)));
    k = log(exp(h) / (((r_bar + delta) / alpha)^(1 / (1 - alpha))));
    y = log((exp(k)^alpha) * (exp(h)^(1 - alpha)));
    i = log(delta * exp(k));
    c = 0.110602;
    lambda = log((exp(c) - (exp(h)^omega) / omega)^(-gamma));
    psi_4 = exp(lambda);
    tb_y = 1 - ((exp(c) + exp(i)) / exp(y));
    util = (((exp(c) - omega^(-1) * exp(h)^omega)^(1 - gamma)) - 1) / (1 - gamma);
    a = 0;
end;
@#endif

@#if model5 == 1
model;
    d = (1 + exp(r(-1))) * d(-1) - exp(y) + exp(c) + exp(i) + (phi / 2) * (exp(k) - exp(k(-1)))^2;
    exp(y) = exp(a) * (exp(k(-1))^alpha) * (exp(h)^(1 - alpha));
    exp(k) = exp(i) + (1 - delta) * exp(k(-1));
    exp(lambda) = beta * (1 + exp(r)) * exp(lambda(+1));
    (exp(c) - (exp(h)^omega) / omega)^(-gamma) = exp(lambda);
    ((exp(c) - (exp(h)^omega) / omega)^(-gamma)) * (exp(h)^(omega - 1)) = exp(lambda) * (1 - alpha) * exp(y) / exp(h);
    exp(lambda) * (1 + phi * (exp(k) - exp(k(-1)))) = beta * exp(lambda(+1)) * (alpha * exp(y(+1)) / exp(k) + 1 - delta + phi * (exp(k(+1)) - exp(k)));
    a = rho * a(-1) + sigma_tfp * e;
    exp(r) = r_bar;
    tb_y = 1 - ((exp(c) + exp(i) + (phi / 2) * (exp(k) - exp(k(-1)))^2) / exp(y));
    ca_y = (1 / exp(y)) * (d(-1) - d);
    util = (((exp(c) - omega^(-1) * exp(h)^omega)^(1 - gamma)) - 1) / (1 - gamma);
end;

steady_state_model;
    beta = 1 / (1 + r_bar);
    r = log((1 - beta) / beta);
    d = d_bar;
    h = log(((1 - alpha) * (alpha / (r_bar + delta))^(alpha / (1 - alpha)))^(1 / (omega - 1)));
    k = log(exp(h) / (((r_bar + delta) / alpha)^(1 / (1 - alpha))));
    y = log((exp(k)^alpha) * (exp(h)^(1 - alpha)));
    i = log(delta * exp(k));
    c = log(exp(y) - exp(i) - r_bar * d);
    tb_y = 1 - ((exp(c) + exp(i)) / exp(y));
    util = (((exp(c) - omega^(-1) * exp(h)^omega)^(1 - gamma)) - 1) / (1 - gamma);
    lambda = log((exp(c) - (exp(h)^omega) / omega)^(-gamma));
    a = 0;
    ca_y = 0;
end;
@#endif

check;
steady;

shocks;
    var e; stderr 1;
end;

stoch_simul(order=1, irf=0, periods=@{periods}, nomoments, nofunctions, nograph, nocorr, noprint);
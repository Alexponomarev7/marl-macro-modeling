@#ifndef periods
    @#define periods = 100
@#endif

var w           $W$         (long_name='Real Wage')
    r           $r$         (long_name='Real Return On Capital')
    c           $C$         (long_name='Real Consumption')
    k           $K$         (long_name='Capital Stock')
    h           $H$         (long_name='Hours Worked')
    m           $M$         (long_name='Money Stock')
    p           $P$         (long_name='Price Level')
    pstar       ${P^*}$     (long_name='Foreign Price Level')
    g           $g$         (long_name='Growth Rate Of Money Stock')
    lambda      $\lambda$   (long_name='Total Factor Productivity')
    b           $B$         (long_name='Foreign Bonds')
    rf          ${r^f}$     (long_name='Foreign Interest Rate')
    e           $e$         (long_name='Exchange Rate')
    x           $X$         (long_name='Net Exports');

varexo eps_lambda       ${\varepsilon^\lambda}$ (long_name='TFP Shock')
       eps_g            ${\varepsilon^g}$       (long_name='Money Growth Shock')
       eps_pstar        ${\varepsilon^*}$       (long_name='Foreign Price Level Shock');

parameters beta         ${\beta}$    (long_name='Discount Factor')
           delta        ${\delta}$   (long_name='Depreciation Rate')
           theta        ${\theta}$   (long_name='Capital Share Production')
           kappa        ${\kappa}$   (long_name='Capital Adjustment Cost')
           a            ${a}$        (long_name='Risk Premium')
           B            ${B}$        (long_name='Composite Labor Disutility Parameter')
           gamma_lambda ${\gamma_\lambda}$ (long_name='Autocorrelation TFP')
           gamma_g      ${\gamma_g}$ (long_name='Autocorrelation Money Growth')
           gamma_pstar  ${\gamma_{P^*}}$ (long_name='Autocorrelation Foreign Price')
           pistar       ${\pi^*}$    (long_name='Foreign Inflation')
           rstar        ${\r^*}$     (long_name='Foreign Interest Rate')
           sigma_lambda ${\sigma_\lambda}$ (long_name='Standard Deviation TFP Shock')
           sigma_g      ${\sigma_g}$ (long_name='Standard Deviation Money Shock')
           sigma_pstar  ${\sigma_{P^*}}$ (long_name='Standard Deviation Foreign Price Shock');

kappa = 0.5;
beta = 0.99;
delta = 0.025;
theta = 0.36;
rstar = 0.03;
a = 0.01;
B = -2.58;
gamma_lambda = 0.95;
gamma_g = 0.95;
gamma_pstar = 0.95;
sigma_lambda = 0.01;
sigma_g = 0.01;
sigma_pstar = 0.01;

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
    g = (1 - gamma_g) * 1 + gamma_g * g(-1) + sigma_g * eps_g;
    pstar = (1 - gamma_pstar) * 1 + gamma_pstar * pstar(-1) + sigma_pstar * eps_pstar;
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
    var eps_lambda; stderr 1;
    var eps_g; stderr 1;
    var eps_pstar; stderr 1;
end;

steady;

stoch_simul(order=1, irf=100, periods=@{periods}, nomoments, nofunctions, nograph, nocorr, noprint);
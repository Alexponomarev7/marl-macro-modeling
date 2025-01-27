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
    g           $g$         (long_name='Growth Rate Of Money Stock')
    lambda      $\lambda$   (long_name='Total Factor Productivity')
    y           $y$         (long_name='Real Output');

varexo eps_lambda       ${\varepsilon^\lambda}$ (long_name='TFP Shock')
       eps_g            ${\varepsilon^g}$       (long_name='Money Growth Shock');

parameters beta         ${\beta}$    (long_name='Discount Factor')
           delta        ${\delta}$   (long_name='Depreciation Rate')
           theta        ${\theta}$   (long_name='Capital Share Production')
           A            ${A}$        (long_name='Labor Disutility Parameter')
           h_0          ${h_0}$      (long_name='Steady State Hours Worked')
           B            ${B}$        (long_name='Composite Labor Disutility Parameter')
           gamma        ${\gamma}$   (long_name='Autocorrelation TFP')
           pi           ${\pi}$      (long_name='Autocorrelation Money Growth')
           g_bar        ${\bar g}$   (long_name='Steady State Growth Rate Of Money')
           D            ${D}$        (long_name='Coefficient Log Balances');

predetermined_variables k;

beta = 0.99;
delta = 0.025;
theta = 0.36;
A = 1.72;
h_0 = 0.583;
gamma = 0.95;
pi = 0.48;
g_bar = 1;
D = 0.01;

model;
    c + k(+1) + m / p = w * h + r * k + (1 - delta) * k + m(-1) / p + (g - 1) * m(-1) / p;
    1 / c = beta * p / (c(+1) * p(+1)) + D * p / m;
    1 / c = -B / w;
    1 / c = beta / c(+1) * (r(+1) + 1 - delta);
    m = g * m(-1);
    y = lambda * k^theta * h^(1 - theta);
    w = (1 - theta) * lambda * k^theta * h^(-theta);
    r = theta * lambda * (k / h)^(theta - 1);
    log(g) = (1 - pi) * log(g_bar) + pi * log(g(-1)) + eps_g;
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

steady;

shocks;
    var eps_g; stderr 0.01;
end;

stoch_simul(irf=100, order=1) k c w r h m y g p;

shocks(overwrite);
    var eps_lambda; stderr 0.01;
end;

stoch_simul(irf=100, order=1, periods=@{periods}, nomoments, nofunctions, nograph, nocorr, noprint);
@#ifndef periods
    @#define periods = 100
@#endif

@#define indivisible_labor = 1

@#ifndef beta
    @#define beta = 0.99
@#endif
@#ifndef delta
    @#define delta = 0.025
@#endif
@#ifndef theta
    @#define theta = 0.36
@#endif
@#ifndef gammma
    @#define gammma = 0.95
@#endif
@#ifndef A
    @#define A = 2
@#endif
@#ifndef sigma_eps
    @#define sigma_eps = 0.00712
@#endif
@#ifndef h_0
    @#define h_0 = 0.53
@#endif

var c $c$                  (long_name='Consumption')
    w $w$                  (long_name='Real Wage')
    r $r$                  (long_name='Real Interest Rate')
    y $y$                  (long_name='Output')
    h $h$                  (long_name='Hours Worked')
    k $k$                  (long_name='Capital Stock')
    invest $i$             (long_name='Investment')
    lambda $\lambda$       (long_name='Total Factor Productivity')
    productivity ${\frac{y}{h}}$ (long_name='Productivity');

varexo eps_a ${\varepsilon_a}$ (long_name='TFP Shock');

parameters beta $\beta$    (long_name='Discount Factor')
           delta $\delta$  (long_name='Depreciation Rate')
           theta $\theta$  (long_name='Capital Share')
           gamma $\gamma$  (long_name='AR Coefficient TFP')
           A $A$           (long_name='Labor Disutility Parameter')
           h_0 ${h_0}$     (long_name='Full Time Workers In Steady State')
           sigma_eps $\sigma_e$ (long_name='TFP Shock Volatility')
           B $B$           (long_name='Composite Labor Disutility Parameter');

beta = @{beta};
delta = @{delta};
theta = @{theta};
gammma = @{gammma};
A = @{A};
sigma_eps = @{sigma_eps};
h_0 = @{h_0};

model;
    1 / c = beta * ((1 / c(+1)) * (r(+1) + (1 - delta)));
    @#if indivisible_labor
        (1 - theta) * (y / h) = B * c;
    @#else
        (1 - theta) * (y / h) = A / (1 - h) * c;
    @#endif
    c = y + (1 - delta) * k(-1) - k;
    k = (1 - delta) * k(-1) + invest;
    y = lambda * k(-1)^theta * h^(1 - theta);
    r = theta * (y / k(-1));
    w = (1 - theta) * (y / h);
    log(lambda) = gamma * log(lambda(-1)) + eps_a;
    productivity = y / h;
end;

steady_state_model;
    B = -A * (log(1 - h_0)) / h_0;
    lambda = 1;
    @#if indivisible_labor
        h = (1 - theta) * (1 / beta - (1 - delta)) / (B * (1 / beta - (1 - delta) - theta * delta));
    @#else
        h = (1 + (A / (1 - theta)) * (1 - (beta * delta * theta) / (1 - beta * (1 - delta))))^(-1);
    @#endif
    k = h * ((1 / beta - (1 - delta)) / (theta * lambda))^(1 / (theta - 1));
    invest = delta * k;
    y = lambda * k^theta * h^(1 - theta);
    c = y - delta * k;
    r = 1 / beta - (1 - delta);
    w = (1 - theta) * (y / h);
    productivity = y / h;
end;

steady;

shocks;
    var eps_a; stderr sigma_eps;
end;

check;

stoch_simul(order=1, irf=20, hp_filter=1600, periods=@{periods}, loglinear, nomoments, nofunctions, nograph, nocorr, noprint);
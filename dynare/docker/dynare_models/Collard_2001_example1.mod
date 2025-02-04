@#ifndef periods
    @#define periods = 100
@#endif

@#ifndef alpha
    @#define alpha = 0.36
@#endif
@#ifndef rho
    @#define rho = 0.95
@#endif
@#ifndef tau
    @#define tau = 0.025
@#endif
@#ifndef beta
    @#define beta = 0.99
@#endif
@#ifndef delta
    @#define delta = 0.025
@#endif
@#ifndef psi
    @#define psi = 0
@#endif
@#ifndef theta
    @#define theta = 2.95
@#endif
@#ifndef phi
    @#define phi = 0.1
@#endif

var y $y$                              (long_name='Output')
    c $c$                              (long_name='Consumption')
    k $k$                              (long_name='Capital')
    a $a$                              (long_name='Technology Shock')
    h $h$                              (long_name='Labor')
    b $b$                              (long_name='Preference Shock');

varexo e $\varepsilon$                 (long_name='Technology Shock Innovation')
       u $u$                           (long_name='Preference Shock Innovation');

parameters beta $\beta$                (long_name='Discount Factor')
           rho $\rho$                  (long_name='Persistence Of Shocks')
           alpha $\alpha$              (long_name='Capital Share')
           delta $\delta$              (long_name='Depreciation Rate')
           theta $\theta$              (long_name='Relative Risk Aversion')
           psi $\psi$                  (long_name='Inverse Frisch Elasticity')
           tau $\tau$                  (long_name='Spillover Between Shocks')
           phi $\phi$                  (long_name='Correlation Between Shocks');

alpha = @{alpha};
rho   = @{rho};
tau   = @{tau};
beta  = @{beta};
delta = @{delta};
psi   = @{psi};
theta = @{theta};
phi   = @{phi};

model;
    c * theta * h^(1 + psi) = (1 - alpha) * y;
    k = beta * (((exp(b) * c) / (exp(b(+1)) * c(+1)))
        * (exp(b(+1)) * alpha * y(+1) + (1 - delta) * k));
    y = exp(a) * (k(-1)^alpha) * (h^(1 - alpha));
    k = exp(b) * (y - c) + (1 - delta) * k(-1);
    a = rho * a(-1) + tau * b(-1) + e;
    b = tau * a(-1) + rho * b(-1) + u;
end;

initval;
    y = 1.08068253095672;
    c = 0.80359242014163;
    h = 0.29175631001732;
    k = 11.08360443260358;
    a = 0;
    b = 0;
    e = 0;
    u = 0;
end;

shocks;
    var e; stderr 0.009;
    var u; stderr 0.009;
    var e, u = phi * 0.009 * 0.009;
end;

steady;

stoch_simul(order=1, irf=0, periods=@{periods}, nomoments, nofunctions, nograph, nocorr, noprint);
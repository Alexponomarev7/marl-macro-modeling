@#ifndef periods
    @#define periods = 100
@#endif

@#define extreme_calibration = 1

var V $V$                              (long_name='Value Function')
    y $y$                              (long_name='Output')
    c $c$                              (long_name='Consumption')
    k $k$                              (long_name='Capital')
    invest $i$                         (long_name='Investment')
    l $l$                              (long_name='Labor')
    z $z$                              (long_name='Technology Shock')
    s $s$                              (long_name='Auxiliary Variable For Value Function')
    E_t_SDF_plus_1 ${E_t(SDF_{t+1})}$  (long_name='Expected Stochastic Discount Factor')
    sigma $\sigma$                     (long_name='Volatility')
    E_t_R_k ${E_t(R^k_{t+1})}$         (long_name='Expected Return On Capital')
    R_f ${R^f}$                        (long_name='Risk-Free Rate');

varexo e $\varepsilon$                 (long_name='Technology Shock Innovation')
       omega $\omega$                  (long_name='Volatility Shock');

parameters beta $\beta$                (long_name='Discount Factor')
           gamma $\gamma$              (long_name='Risk Aversion')
           delta $\delta$              (long_name='Depreciation Rate')
           nu $\nu$                    (long_name='Consumption Utility Weight')
           psi $\psi$                  (long_name='Elasticity Of Intertemporal Substitution')
           lambda $\lambda$            (long_name='Persistence Of Technology Shock')
           zeta $\zeta$                (long_name='Capital Share')
           rho $\rho$                  (long_name='Persistence Of Volatility')
           sigma_bar ${\bar \sigma}$   (long_name='Steady State Volatility')
           eta $\eta$                  (long_name='Volatility Shock Scale');

beta = @{beta};
nu = @{nu};
zeta = @{zeta};
delta = @{delta};
lambda = @{delta};

@#if extreme_calibration
    psi = 0.5;
    gamma = 40;
    sigma_bar = log(0.021); % typo in paper; not log(sigma)=0.007
    eta = 0.1;
@#else
    psi = 0.5;
    gamma = 5;
    sigma_bar = log(0.007); % typo in paper; not log(sigma)=exp0.007
    eta = 0.06;
@#endif

rho = 0.9;

model;
    #theta = (1 - gamma) / (1 - (1 / psi));

    // Define Value function
    V = ((1 - beta) * ((c^nu) * ((1 - l)^(1 - nu)))^((1 - gamma) / theta) + beta * s^(1 / theta))^(theta / (1 - gamma));

    // Define an auxiliary variable s that captures E_t[V(+1)^sigma]
    s = V(+1)^(1 - gamma);

    // Euler equation: was wrong in paper as labor term was missing
    1 = beta * (((1 - l(+1)) / (1 - l))^(1 - nu) * (c(+1) / c)^nu)^((1 - gamma) / theta) * c / c(+1)
        * ((V(+1)^(1 - gamma)) / s)^(1 - (1 / theta)) * (zeta * exp(z(+1)) * k^(zeta - 1) * l(+1)^(1 - zeta) + 1 - delta);

    // Define net return to capital
    E_t_R_k = zeta * exp(z(+1)) * k^(zeta - 1) * l(+1)^(1 - zeta) - delta;

    // Define expected value of stochastic discount factor
    E_t_SDF_plus_1 = beta * (((1 - l(+1)) / (1 - l))^(1 - nu) * (c(+1) / c)^nu)^((1 - gamma) / theta) * c / c(+1)
        * ((V(+1)^(1 - gamma)) / s)^(1 - (1 / theta));

    // Define net risk-free rate
    R_f = (1 / E_t_SDF_plus_1 - 1);

    // Labor supply FOC
    ((1 - nu) / nu) * (c / (1 - l)) = (1 - zeta) * exp(z) * (k(-1)^(zeta)) * (l^(-zeta));

    // Budget constraint
    c + invest = exp(z) * (k(-1)^(zeta)) * (l^(1 - zeta));

    // Law of motion of capital
    k = (1 - delta) * k(-1) + invest;

    // Technology shock
    z = lambda * z(-1) + exp(sigma) * e;

    // Output definition
    y = exp(z) * (k(-1)^(zeta)) * (l^(1 - zeta));

    // Law of motion of volatility
    sigma = (1 - rho) * sigma_bar + rho * sigma(-1) + eta * omega;
end;

steady_state_model;
    %% Steady state actually used; sets labor to 1/3 and adjusts nu accordingly
    l = 1 / 3;
    k = ((1 - beta * (1 - delta)) / (zeta * beta))^(1 / (zeta - 1)) * l;
    c = k^zeta * l^(1 - zeta) - delta * k;
    nu = c / ((1 - zeta) * k^zeta * l^(-zeta) * (1 - l) + c);
    invest = k^zeta * l^(1 - zeta) - c;
    V = c^nu * (1 - l)^(1 - nu);
    s = V^(1 - gamma);
    z = 0;
    y = k^zeta * l^(1 - zeta);
    E_t_SDF_plus_1 = beta;
    sigma = sigma_bar;
    E_t_R_k = zeta * k^(zeta - 1) * l^(1 - zeta) - delta;
    R_f = 1 / E_t_SDF_plus_1 - 1;
end;

steady;
check;

shocks;
    var e; stderr 1;
    var omega; stderr 1;
end;

stoch_simul(order=2, irf=0, periods=@{periods}, nomoments, nofunctions, nograph, nocorr, noprint);
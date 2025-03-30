var V                               (long_name='Value Function')
    y                               (long_name='Output')
    c                               (long_name='Consumption')
    k                               (long_name='Capital')
    invest                          (long_name='Investment')
    l                               (long_name='Labor')
    z                               (long_name='Technology Shock')
    s                               (long_name='Auxiliary Variable For Value Function')
    E_t_SDF_plus_1                  (long_name='Expected Stochastic Discount Factor')
    sigma                           (long_name='Volatility')
    E_t_R_k                         (long_name='Expected Return On Capital')
    R_f                             (long_name='Risk-Free Rate');

varexo e                (long_name='Technology Shock Innovation');

parameters beta                  (long_name='Discount Factor')
           gammma                (long_name='Risk Aversion')
           delta                 (long_name='Depreciation Rate')
           nu                    (long_name='Consumption Utility Weight')
           psi                   (long_name='Elasticity Of Intertemporal Substitution')
           lambda                (long_name='Persistence Of Technology Shock')
           zeta                  (long_name='Capital Share')
           rho                   (long_name='Persistence Of Volatility')
           sigma_bar             (long_name='Steady State Volatility')
           eta                   (long_name='Volatility Shock Scale');

beta = {beta};
nu = {nu};
zeta = {zeta};
delta = {delta};
lambda = {lambda};
psi = {psi};
gammma = {gammma};
sigma_bar = {sigma_bar};
eta = {eta};
rho = {rho};

model;
    #theta = (1 - gammma) / (1 - (1 / psi));

    // Define Value function
    V = ((1 - beta) * ((c^nu) * ((1 - l)^(1 - nu)))^((1 - gammma) / theta) + beta * s^(1 / theta))^(theta / (1 - gammma));

    // Define an auxiliary variable s that captures E_t[V(+1)^sigma]
    s = V(+1)^(1 - gammma);

    // Euler equation: was wrong in paper as labor term was missing
    1 = beta * (((1 - l(+1)) / (1 - l))^(1 - nu) * (c(+1) / c)^nu)^((1 - gammma) / theta) * c / c(+1)
        * ((V(+1)^(1 - gammma)) / s)^(1 - (1 / theta)) * (zeta * exp(z(+1)) * k^(zeta - 1) * l(+1)^(1 - zeta) + 1 - delta);

    // Define net return to capital
    E_t_R_k = zeta * exp(z(+1)) * k^(zeta - 1) * l(+1)^(1 - zeta) - delta;

    // Define expected value of stochastic discount factor
    E_t_SDF_plus_1 = beta * (((1 - l(+1)) / (1 - l))^(1 - nu) * (c(+1) / c)^nu)^((1 - gammma) / theta) * c / c(+1)
        * ((V(+1)^(1 - gammma)) / s)^(1 - (1 / theta));

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
    sigma = (1 - rho) * sigma_bar + rho * sigma(-1) + eta;
end;

steady_state_model;
    %% Steady state actually used; sets labor to 1/3 and adjusts nu accordingly
    l = 1 / 3;
    k = ((1 - beta * (1 - delta)) / (zeta * beta))^(1 / (zeta - 1)) * l;
    c = k^zeta * l^(1 - zeta) - delta * k;
    nu = c / ((1 - zeta) * k^zeta * l^(-zeta) * (1 - l) + c);
    invest = k^zeta * l^(1 - zeta) - c;
    V = c^nu * (1 - l)^(1 - nu);
    s = V^(1 - gammma);
    z = 0;
    y = k^zeta * l^(1 - zeta);
    E_t_SDF_plus_1 = beta;
    sigma = sigma_bar;
    E_t_R_k = zeta * k^(zeta - 1) * l^(1 - zeta) - delta;
    R_f = 1 / E_t_SDF_plus_1 - 1;
end;

shocks;
    var e; periods {shock_periods}; values {shock_values};
end;

steady;
check;

perfect_foresight_setup(periods={periods});
perfect_foresight_solver;
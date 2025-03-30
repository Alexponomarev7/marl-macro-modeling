var y                               (long_name='Output')
    c                               (long_name='Consumption')
    k                               (long_name='Capital')
    a                               (long_name='Technology Shock')
    h                               (long_name='Labor')
    b                               (long_name='Preference Shock');

varexo e                (long_name='Technology Shock Innovation');

parameters beta                  (long_name='Discount Factor')
           rho                   (long_name='Persistence Of Shocks')
           alpha                 (long_name='Capital Share')
           delta                 (long_name='Depreciation Rate')
           theta                 (long_name='Relative Risk Aversion')
           psi                   (long_name='Inverse Frisch Elasticity')
           tau                   (long_name='Spillover Between Shocks')
           phi                   (long_name='Correlation Between Shocks');

alpha = {alpha};
rho   = {rho};
tau   = {tau};
beta  = {beta};
delta = {delta};
psi   = {psi};
theta = {theta};
phi   = {phi};

model;
    c * theta * h^(1 + psi) = (1 - alpha) * y;
    k = beta * (((exp(b) * c) / (exp(b(+1)) * c(+1)))
        * (exp(b(+1)) * alpha * y(+1) + (1 - delta) * k));
    y = exp(a) * (k(-1)^alpha) * (h^(1 - alpha));
    k = exp(b) * (y - c) + (1 - delta) * k(-1);
    a = rho * a(-1) + tau * b(-1) + e;
    b = tau * a(-1) + rho * b(-1);
end;

initval;
    y = 1.08068253095672;
    c = 0.80359242014163;
    h = 0.29175631001732;
    k = 11.08360443260358;
    a = 0;
    b = 0;
    e = 0;
end;

shocks;
    var e; periods {shock_periods}; values {shock_values};
end;

steady;
check;

perfect_foresight_setup(periods={periods});
perfect_foresight_solver;
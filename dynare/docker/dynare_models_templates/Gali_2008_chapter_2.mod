var C                  (long_name='Consumption')
    W_real             (long_name='Real Wage')
    Pi                 (long_name='Inflation')
    A                  (long_name='AR(1) Technology Process')
    N                  (long_name='Hours Worked')
    R                  (long_name='Nominal Interest Rate')
    realinterest       (long_name='Real Interest Rate')
    Y                  (long_name='Output')
    m_growth_ann       (long_name='Money Growth')
;

varexo eps_A           (long_name='Technology Shock');

parameters alppha      (long_name='Capital Share')
           betta       (long_name='Discount Factor')
           rho         (long_name='Autocorrelation Technology Shock')
           siggma      (long_name='Log Utility')
           phi         (long_name='Unitary Frisch Elasticity')
           phi_pi      (long_name='Inflation Feedback Taylor Rule')
           eta         (long_name='Semi-Elasticity Of Money Demand')
;

alppha = {alppha};
betta = {betta};
rho = {rho};
siggma = {siggma};
phi = {phi};
phi_pi = {phi_pi};
eta = {eta};

model;
    W_real = C^siggma * N^phi;
    1 / R = betta * (C(+1) / C)^(-siggma) / Pi(+1);
    A * N^(1 - alppha) = C;
    W_real = (1 - alppha) * A * N^(-alppha);
    realinterest = R / Pi(+1);
    R = 1 / betta * Pi^phi_pi;
    C = Y;
    log(A) = rho * log(A(-1)) + eps_A;
    m_growth_ann = 4 * (log(Y) - log(Y(-1)) - eta * (log(R) - log(R(-1))) + log(Pi));
end;

shocks;
    var eps_A; periods {shock_periods}; values {shock_values};
end;

steady_state_model;
    A = 1;
    R = 1 / betta;
    Pi = 1;
    realinterest = R;
    N = (1 - alppha)^(1 / ((1 - siggma) * alppha + phi + siggma));
    C = A * N^(1 - alppha);
    W_real = (1 - alppha) * A * N^(-alppha);
    Y = C;
    m_growth_ann = 0;
end;

steady;
check;

perfect_foresight_setup(periods={periods});
perfect_foresight_solver;
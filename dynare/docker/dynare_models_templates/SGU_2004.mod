var c  (long_name='Consumption')
    k  (long_name='Capital')
    a  (long_name='Technology Shock');

varexo epsilon (long_name='Technology Shock Innovation');

predetermined_variables k;

parameters SIG    (long_name='Intertemporal Elasticity Of Substitution')
           DELTA  (long_name='Depreciation Rate')
           ALFA   (long_name='Capital Share')
           BETTA  (long_name='Discount Factor')
           RHO    (long_name='Persistence Of Technology Shock');

BETTA = {BETTA};
DELTA = {DELTA};
ALFA = {ALFA};
RHO = {RHO};
SIG = {SIG};

model;
    0 = exp(c) + exp(k(+1)) - (1 - DELTA) * exp(k) - exp(a) * exp(k)^ALFA;
    0 = exp(c)^(-SIG) - BETTA * exp(c(+1))^(-SIG) * (exp(a(+1)) * ALFA * exp(k(+1))^(ALFA - 1) + 1 - DELTA);
    0 = a - RHO * a(-1) - epsilon;
end;

steady_state_model;
    k = log(((1 / BETTA + DELTA - 1) / ALFA)^(1 / (ALFA - 1)));
    c = log(exp(k)^ALFA - DELTA * exp(k));
    a = 0;
end;

shocks;
    var epsilon; periods {shock_periods}; values {shock_values};
end;

steady;
check;

perfect_foresight_setup(periods={periods});
perfect_foresight_solver;
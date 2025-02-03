@#ifndef periods
    @#define periods = 100
@#endif

var c ${c}$ (long_name='Consumption')
    k ${k}$ (long_name='Capital')
    a ${a}$ (long_name='Technology Shock');

varexo epsilon ${\varepsilon}$ (long_name='Technology Shock Innovation');

predetermined_variables k;

parameters SIG ${\sigma}$ (long_name='Intertemporal Elasticity Of Substitution')
           DELTA ${\delta}$ (long_name='Depreciation Rate')
           ALFA ${\alpha}$ (long_name='Capital Share')
           BETTA ${\beta}$ (long_name='Discount Factor')
           RHO ${\rho}$ (long_name='Persistence Of Technology Shock');

BETTA = 0.95;
DELTA = 1;
ALFA = 0.3;
RHO = 0;
SIG = 2;

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
    var epsilon; stderr 1;
end;

steady;
check;

stoch_simul(order=2, periods=@{periods}, nomoments, nofunctions, nograph, nocorr, noprint);
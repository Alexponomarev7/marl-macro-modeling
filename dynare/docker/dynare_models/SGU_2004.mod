@#ifndef periods
    @#define periods = 100
@#endif

var c ${c}$ (long_name='consumption (log)')
    k ${k}$ (long_name='capital (log)')
    a ${a}$ (long_name='Total Factor Productivity');   % log TFP (enters as exp(a))

varexo epsilon ${\varepsilon}$ (long_name='Technology Shock Innovation');

predetermined_variables k;

parameters SIG ${\sigma}$ (long_name='Intertemporal Elasticity Of Substitution')
           DELTA ${\delta}$ (long_name='Depreciation Rate')
           ALFA ${\alpha}$ (long_name='Capital Share')
           BETTA ${\beta}$ (long_name='Discount Factor')
           RHO ${\rho}$ (long_name='Persistence Of Technology Shock');

% DELTA stays 1 (full depreciation, the paper's closed-form benchmark)
@#if !defined(BETTA)
    @#define BETTA = 0.95
@#endif
@#if !defined(ALFA)
    @#define ALFA = 0.3
@#endif
@#if !defined(RHO)
    @#define RHO = 0
@#endif
@#if !defined(SIG)
    @#define SIG = 2
@#endif
@#if !defined(technology_shock_stderr)
    @#define technology_shock_stderr = 0.01
@#endif

BETTA = @{BETTA};
DELTA = 1;
ALFA = @{ALFA};
RHO = @{RHO};
SIG = @{SIG};

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
    var epsilon; stderr @{technology_shock_stderr};
end;

steady;
check;

stoch_simul(order=2, periods=@{periods}, nomoments, nofunctions, nograph, nocorr, noprint);
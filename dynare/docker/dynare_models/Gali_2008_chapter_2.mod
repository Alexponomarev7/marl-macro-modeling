@#ifndef periods
    @#define periods = 100
@#endif

var C ${C}$                  (long_name='Consumption')
    W_real ${\frac{W}{P}}$   (long_name='Real Wage')
    Pi ${\Pi}$               (long_name='Inflation')
    A ${A}$                  (long_name='AR(1) Technology Process')
    N ${N}$                  (long_name='Hours Worked')
    R ${R^n}$                (long_name='Nominal Interest Rate')
    realinterest ${R^{r}}$   (long_name='Real Interest Rate')
    Y ${Y}$                  (long_name='Output')
    m_growth_ann ${\Delta M}$ (long_name='Money Growth');

varexo eps_A ${\varepsilon_A}$ (long_name='Technology Shock')
       eps_m ${\varepsilon_m}$ (long_name='Monetary Policy Shock');

parameters alppha ${\alpha}$ (long_name='Capital Share')
           betta ${\beta}$   (long_name='Discount Factor')
           rho ${\rho}$      (long_name='Autocorrelation Technology Shock')
           siggma ${\sigma}$ (long_name='Log Utility')
           phi ${\phi}$      (long_name='Unitary Frisch Elasticity')
           phi_pi ${\phi_{\pi}}$ (long_name='Inflation Feedback Taylor Rule')
           eta ${\eta}$      (long_name='Semi-Elasticity Of Money Demand');

@#if !defined(alppha)
    @#define alppha = 0.33
@#endif
@#if !defined(betta)
    @#define betta = 0.99
@#endif
@#if !defined(rho)
    @#define rho = 0.9
@#endif
@#if !defined(siggma)
    @#define siggma = 1
@#endif
@#if !defined(phi)
    @#define phi = 1
@#endif
@#if !defined(phi_pi)
    @#define phi_pi = 1.5
@#endif
@#if !defined(eta)
    @#define eta = 4
@#endif
@#if !defined(technology_shock_stderr)
    @#define technology_shock_stderr = 0.01
@#endif
@#if !defined(monetary_shock_stderr)
    @#define monetary_shock_stderr = 0.0025
@#endif

alppha = @{alppha};
betta = @{betta};
rho = @{rho};
siggma = @{siggma};
phi = @{phi};
phi_pi = @{phi_pi};
eta = @{eta};

model;
    W_real = C^siggma * N^phi;
    1 / R = betta * (C(+1) / C)^(-siggma) / Pi(+1);
    A * N^(1 - alppha) = C;
    W_real = (1 - alppha) * A * N^(-alppha);
    realinterest = R / Pi(+1);
    R = 1 / betta * Pi^phi_pi + eps_m;
    C = Y;
    log(A) = rho * log(A(-1)) + eps_A;
    m_growth_ann = 4 * (log(Y) - log(Y(-1)) - eta * (log(R) - log(R(-1))) + log(Pi));
end;

shocks;
    var eps_A; stderr @{technology_shock_stderr};
    var eps_m; stderr @{monetary_shock_stderr};
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

stoch_simul(irf=20, order=1, periods=@{periods}, nomoments, nofunctions, nograph, nocorr, noprint);
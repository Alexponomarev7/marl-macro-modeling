@#ifndef periods
    @#define periods = 100
@#endif

var C $Consumption$                         (long_name='Consumption')
    W_real $Real Wage$                      (long_name='Real Wage')
    Pi $Inflation$                          (long_name='Inflation')
    LoggedProductivity $LoggedProductivity$ (long_name='AR(1) Productivity Process')
    N $HoursWorked$                         (long_name='Hours Worked')
    R $Nominal Interest Rate$               (long_name='Nominal Interest Rate')
    r $Real Interest Rate$                  (long_name='Real Interest Rate')
    Y $Output$                              (long_name='Output')
    m_growth_ann $Money Growth$             (long_name='Money Growth');

varexo eps_A $Technology Shock$ (long_name='Technology Shock')
       eps_m $Monetary Policy Shock$ (long_name='Monetary Policy Shock');

parameters alppha ${\alpha}$ (long_name='Capital Share')
           betta ${\beta}$   (long_name='Discount Factor')
           rho ${\rho}$      (long_name='Autocorrelation Technology Shock')
           siggma ${\sigma}$ (long_name='Log Utility')
           phi ${\phi}$      (long_name='Unitary Frisch Elasticity')
           phi_pi ${\phi_{\pi}}$ (long_name='Inflation Feedback Taylor Rule')
           eta ${\eta}$      (long_name='Semi-Elasticity Of Money Demand');

alppha = 0.33;
betta = 0.99;
rho = 0.9;
siggma = 1;
phi = 1;
phi_pi = 1.5;
eta = 4;

model;
    W_real = C^siggma * N^phi;
    1 / R = betta * (C(+1) / C)^(-siggma) / Pi(+1);
    exp(LoggedProductivity) * N^(1 - alppha) = C;
    W_real = (1 - alppha) * exp(LoggedProductivity) * N^(-alppha);
    r = R / Pi(+1);
    R = 1 / betta * Pi^phi_pi + eps_m;
    C = Y;
    LoggedProductivity = rho * LoggedProductivity(-1) + eps_A;
    m_growth_ann = 4 * (log(Y) - log(Y(-1)) - eta * (log(R) - log(R(-1))) + log(Pi));
end;

shocks;
    var eps_A; stderr 0.5;
end;

steady_state_model;
    LoggedProductivity = 0;
    R = 1 / betta;
    Pi = 1;
    r = R;
    N = (1 - alppha)^(1 / ((1 - siggma) * alppha + phi + siggma));
    C = exp(LoggedProductivity) * N^(1 - alppha);
    W_real = (1 - alppha) * exp(LoggedProductivity) * N^(-alppha);
    Y = C;
    m_growth_ann = 0;
end;

steady;
check;

stoch_simul(irf=20, order=1, periods=200, nomoments, nofunctions, nograph, nocorr, noprint);

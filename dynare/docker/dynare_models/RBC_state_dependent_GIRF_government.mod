% RBC Model with an optimizing fiscal authority, derived from Pfeifer's RBC_state_dependent_GIRF:
% households value a public good (utility adds chi_t*log(G_t)), and the government sets G by the
% Samuelson rule chi_t / G_t = C_t^(-sigma). chi_bar sets steady-state G/Y = gshare.
%
% Features:
%   - Optimal public-good provision (Samuelson rule), public-good preference shocks
%   - TFP shocks (AR(1) process)
%   - Population and technology growth
%
% Timing convention:
%   - Capital_t is end-of-period t capital
%   - Production at t uses Capital_{t-1}

var Consumption $Consumption$ (long_name='Consumption')
    Capital $Capital$ (long_name='Capital')
    Output $Output$ (long_name='Output')
    Labor $Labor$ (long_name='Hours Worked')
    LoggedProductivity $LoggedProductivity$ (long_name='Total Factor Productivity')
    LoggedGovSpending $LoggedGovSpending$ (long_name='Government Spending')
    PublicGoodPreference $PublicGoodPreference$ (long_name='public good preference (log deviation of chi)')
    InterestRate $InterestRate$ (long_name='Annualized Interest Rate')
    Wage $Wage$ (long_name='Real Wage')
    Investment $Investment$ (long_name='Investment')
    GovSpending $GovSpending$ (long_name='Government Spending Level');

varexo ProductivityShock $ProductivityShock$ (long_name='TFP shock')
       PublicGoodPreferenceShock $PublicGoodPreferenceShock$ (long_name='Public Good Preference Shock');

parameters alpha ${\alpha}$ (long_name='Capital Share')
           beta ${\beta}$ (long_name='Discount Factor')
           delta ${\delta}$ (long_name='Depreciation Rate')
           sigma ${\sigma}$ (long_name='Risk Aversion')
           psi ${\psi}$ (long_name='Labor Disutility Parameter')
           rho ${\rho}$ (long_name='Persistence TFP Shock')
           rho_g ${\rho_g}$ (long_name='Persistence Public Good Preference Shock')
           chi_bar ${\bar\chi}$ (long_name='Public Good Utility Weight')
           n ${n}$ (long_name='Population Growth')
           x ${x}$ (long_name='Technology Growth')
           gammax ${\gamma_x}$ (long_name='Composite Growth Rate')
           gshare ${\frac{G}{Y}}$ (long_name='Government Spending Share')
           l_ss ${l_{ss}}$ (long_name='Steady State Hours Worked')
           k_ss ${k_{ss}}$ (long_name='Steady State Capital')
           i_ss ${i_{ss}}$ (long_name='Steady State Investment')
           y_ss ${y_{ss}}$ (long_name='Steady State Output')
           g_ss ${\bar G}$ (long_name='Steady State Government Spending')
           c_ss ${c_{ss}}$ (long_name='Steady State Consumption')
           k_y ${\frac{K}{Y}}$ (long_name='Capital-Output Ratio')
           i_y ${\frac{I}{Y}}$ (long_name='Investment-Output Ratio');

@#if !defined(alpha)
    @#define alpha = 0.33
@#endif

@#if !defined(sigma)
    @#define sigma = 1.0
@#endif

@#if !defined(n)
    @#define n = 0.0025
@#endif

@#if !defined(x)
    @#define x = 0.005
@#endif

@#if !defined(i_y)
    @#define i_y = 0.22
@#endif

@#if !defined(k_y)
    @#define k_y = 10.0
@#endif

@#if !defined(rho)
    @#define rho = 0.85
@#endif

@#if !defined(rho_g)
    @#define rho_g = 0.95
@#endif

@#if !defined(gshare)
    @#define gshare = 0.20
@#endif

@#if !defined(productivity_shock_stderr)
    @#define productivity_shock_stderr = 0.0068
@#endif

@#if !defined(public_good_preference_shock_stderr)
    @#define public_good_preference_shock_stderr = 0.0105
@#endif

alpha = @{alpha};
sigma = @{sigma};
n = @{n};
x = @{x};
i_y = @{i_y};
k_y = @{k_y};
rho = @{rho};
rho_g = @{rho_g};
gshare = @{gshare};

% Derived parameters
gammax = (1 + n) * (1 + x);
delta = i_y / k_y - x - n - n * x;
beta = gammax / (alpha / k_y + (1 - delta));

l_ss = 0.33;
k_ss = ((1 / beta * gammax - (1 - delta)) / alpha)^(1 / (alpha - 1)) * l_ss;
y_ss = k_ss^alpha * l_ss^(1 - alpha);
i_ss = (x + n + delta + n * x) * k_ss;
g_ss = gshare * y_ss;
c_ss = y_ss - i_ss - g_ss;
w_ss = (1 - alpha) * y_ss / l_ss;
r_ss = 4 * alpha * y_ss / k_ss;
psi = (1 - alpha) * (k_ss / l_ss)^alpha * (1 - l_ss) / c_ss^sigma;
chi_bar = g_ss / c_ss^sigma;  % Samuelson rule at the steady state: g_ss = chi_bar * c_ss^sigma

model;

[name='Labor supply (MRS = MPL)']
psi * Consumption^sigma / (1 - Labor) = (1 - alpha) * exp(LoggedProductivity) * (Capital(-1) / Labor)^alpha;

[name='Euler equation (CRRA)']
Consumption^(-sigma) = beta / gammax * Consumption(+1)^(-sigma) * (alpha * exp(LoggedProductivity(+1)) * (Capital / Labor(+1))^(alpha - 1) + (1 - delta));

[name='Capital accumulation with growth']
gammax * Capital = Output - Consumption - GovSpending + (1 - delta) * Capital(-1);

[name='Production function']
Output = exp(LoggedProductivity) * Capital(-1)^alpha * Labor^(1 - alpha);

[name='TFP law of motion']
LoggedProductivity = rho * LoggedProductivity(-1) + ProductivityShock;

[name='Public good preference law of motion']
PublicGoodPreference = rho_g * PublicGoodPreference(-1) + PublicGoodPreferenceShock;

[name='Optimal public good provision (Samuelson rule)']
GovSpending = chi_bar * exp(PublicGoodPreference) * Consumption^sigma;

[name='Government spending, log deviation from steady state']
LoggedGovSpending = log(GovSpending / g_ss);

[name='Real wage (MPL)']
Wage = (1 - alpha) * Output / Labor;

[name='Annualized interest rate (4 * MPK)']
InterestRate = 4 * alpha * Output / Capital(-1);

[name='Investment definition']
Investment = Output - Consumption - GovSpending;

end;

initval;
  LoggedProductivity = 0;
    LoggedGovSpending = 0;
    PublicGoodPreference = 0;
    Labor = l_ss;
    Capital = k_ss;
    Output = y_ss;
    Investment = i_ss;
    GovSpending = g_ss;
    Consumption = c_ss;
    Wage = w_ss;
    InterestRate = r_ss;
end;

steady;
check;

shocks;
  var ProductivityShock;
  stderr @{productivity_shock_stderr};
  var PublicGoodPreferenceShock;
  stderr @{public_good_preference_shock_stderr};
end;

steady;
check;

stoch_simul(order=1, periods=@{periods}, drop=0, irf=0, nomoments, nofunctions, nograph, nocorr, noprint);

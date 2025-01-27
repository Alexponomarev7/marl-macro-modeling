% Garcia-Cicco, Pancrazi, Uribe (2010) - Real Business Cycles in Emerging Countries
% 
% Features:
%   - Small open economy with debt
%   - Permanent and transitory TFP shocks
%   - Preference shocks
%   - Country premium shocks
%   - Capital adjustment costs
%   - Endogenous interest rate with debt elasticity
%
% Timing convention:
%   - k, d are predetermined (end-of-period t-1, beginning of period t)
%   - Production at t uses k (capital from t-1)

var Consumption             $Consumption$ (long_name='Consumption')
    Capital                 $Capital$ (long_name='Capital') 
    Labor                   $Labor$ (long_name='Hours Worked')
    Output                  $Output$ (long_name='Output')
    Investment              $Investment$ (long_name='Investment')
    Debt                    $Debt$ (long_name='Debt')
    TradeBalance            $TradeBalance$ (long_name='Trade Balance')
    MUConsumption           $MUConsumption$ (long_name='Marginal Utility of Consumption')
    TradeBalanceToOutput    $TradeBalanceToOutput$ (long_name='Trade Balance to Output Ratio')
    OutputGrowth            $OutputGrowth$ (long_name='Output Growth Rate')
    ConsumptionGrowth       $ConsumptionGrowth$ (long_name='Consumption Growth Rate')
    InvestmentGrowth        $InvestmentGrowth$ (long_name='Investment Growth Rate')
    TechGrowthRate          $TechGrowthRate$ (long_name='Technology Growth Rate')
    InterestRate            $InterestRate$ (long_name='Interest Rate')
    LoggedProductivity      $LoggedProductivity$ (long_name='Logged TFP (transitory)')
    CountryPremiumShock     $CountryPremiumShock$ (long_name='Country Premium Shock')
    PreferenceShock         $PreferenceShock$ (long_name='Preference Shock');

predetermined_variables Capital Debt;

varexo ProductivityShock        $\varepsilon_a$ (long_name='Transitory TFP Shock')
       TechGrowthShock          $\varepsilon_g$ (long_name='Permanent TFP Shock')
       PreferenceInnovation     $\varepsilon_\nu$ (long_name='Preference Innovation')
       CountryPremiumInnovation $\varepsilon_\mu$ (long_name='Country Premium Innovation');

parameters beta gamma_c delta alpha psi omega theta phi
           d_ss g_ss
           rho_a rho_g rho_nu rho_mu
           k_ss c_ss y_ss h_ss i_ss r_ss tb_ss mu_c_ss;

@#if !defined(beta)
    @#define beta = 0.9224
@#endif

@#if !defined(gamma_c)
    @#define gamma_c = 2.0
@#endif

@#if !defined(delta)
    @#define delta = 0.1255
@#endif

@#if !defined(alpha)
    @#define alpha = 0.32
@#endif

@#if !defined(psi)
    @#define psi = 0.001
@#endif

@#if !defined(omega)
    @#define omega = 1.6
@#endif

@#if !defined(phi)
    @#define phi = 3.3
@#endif

@#if !defined(d_ss)
    @#define d_ss = 0.007
@#endif

@#if !defined(g_ss)
    @#define g_ss = 1.005
@#endif

@#if !defined(rho_a)
    @#define rho_a = 0.765
@#endif

@#if !defined(rho_g)
    @#define rho_g = 0.828
@#endif

@#if !defined(rho_nu)
    @#define rho_nu = 0.850
@#endif

@#if !defined(rho_mu)
    @#define rho_mu = 0.907
@#endif

@#if !defined(productivity_shock_stderr)
    @#define productivity_shock_stderr = 0.027
@#endif

@#if !defined(tech_growth_shock_stderr)
    @#define tech_growth_shock_stderr = 0.030
@#endif

@#if !defined(preference_shock_stderr)
    @#define preference_shock_stderr = 0.0
@#endif

@#if !defined(country_premium_shock_stderr)
    @#define country_premium_shock_stderr = 0.0
@#endif

alpha = @{alpha};
beta = @{beta};
gamma_c = @{gamma_c};
delta = @{delta};
psi = @{psi};
omega = @{omega};
phi = @{phi};
d_ss = @{d_ss};
g_ss = @{g_ss};
rho_a = @{rho_a};
rho_g = @{rho_g};
rho_nu = @{rho_nu};
rho_mu = @{rho_mu};

% Derived parameter
theta = 1.4 * omega;

r_ss = g_ss^gamma_c / beta;
k_gh_ratio = ((g_ss^gamma_c / beta - 1 + delta) / alpha)^(1 / (alpha - 1));
h_ss = ((1 - alpha) * g_ss * k_gh_ratio^alpha / theta)^(1 / (omega - 1));
k_ss = k_gh_ratio * g_ss * h_ss;
i_ss = (g_ss - 1 + delta) * k_ss;
y_ss = k_ss^alpha * (h_ss * g_ss)^(1 - alpha);
tb_ss = d_ss * (1 - g_ss / r_ss);
c_ss = y_ss - i_ss - tb_ss;
mu_c_ss = (c_ss - theta / omega * h_ss^omega)^(-gamma_c);

model;

[name='Interest rate with debt elasticity and country premium']
InterestRate = g_ss^gamma_c / beta + psi * (exp(Debt - d_ss) - 1) + exp(CountryPremiumShock - 1) - 1;

[name='Marginal utility of consumption (GHH preferences)']
MUConsumption = PreferenceShock * (Consumption - theta / omega * Labor^omega)^(-gamma_c);

[name='Resource constraint']
Output = TradeBalance + Consumption + Investment + phi / 2 * (Capital(+1) / Capital * TechGrowthRate - g_ss)^2 * Capital;

[name='Trade balance definition']
TradeBalance = Debt - Debt(+1) * TechGrowthRate / InterestRate;

[name='Production function']
Output = exp(LoggedProductivity) * Capital^alpha * (TechGrowthRate * Labor)^(1 - alpha);

[name='Investment definition']
Investment = Capital(+1) * TechGrowthRate - (1 - delta) * Capital;

[name='Euler equation']
MUConsumption = beta / TechGrowthRate^gamma_c * InterestRate * MUConsumption(+1);

[name='Labor supply (FOC labor)']
theta * Labor^(omega - 1) = (1 - alpha) * exp(LoggedProductivity) * TechGrowthRate^(1 - alpha) * (Capital / Labor)^alpha;

[name='Investment FOC (Tobin Q)']
MUConsumption * (1 + phi * (Capital(+1) / Capital * TechGrowthRate - g_ss)) 
    = beta / TechGrowthRate^gamma_c * MUConsumption(+1) * (1 - delta 
        + alpha * exp(LoggedProductivity(+1)) * (TechGrowthRate(+1) * Labor(+1) / Capital(+1))^(1 - alpha) 
         + phi * Capital(+2) / Capital(+1) * TechGrowthRate(+1) * (Capital(+2) / Capital(+1) * TechGrowthRate(+1) - g_ss) 
         - phi / 2 * (Capital(+2) / Capital(+1) * TechGrowthRate(+1) - g_ss)^2);

[name='Trade balance to output ratio']
TradeBalanceToOutput = TradeBalance / Output;

[name='Output growth rate']
OutputGrowth = Output / Output(-1) * TechGrowthRate(-1);

[name='Consumption growth rate']
ConsumptionGrowth = Consumption / Consumption(-1) * TechGrowthRate(-1);

[name='Investment growth rate']
InvestmentGrowth = Investment / Investment(-1) * TechGrowthRate(-1);

[name='Transitory TFP process']
LoggedProductivity = rho_a * LoggedProductivity(-1) + ProductivityShock;

[name='Permanent TFP growth process']
log(TechGrowthRate / g_ss) = rho_g * log(TechGrowthRate(-1) / g_ss) + TechGrowthShock;

[name='Preference shock process']
log(PreferenceShock) = rho_nu * log(PreferenceShock(-1)) + PreferenceInnovation;

[name='Country premium shock process']
log(CountryPremiumShock) = rho_mu * log(CountryPremiumShock(-1)) + CountryPremiumInnovation;

end;

steady_state_model;
    InterestRate = r_ss;
    Debt = d_ss;
    Capital = k_ss;
    Labor = h_ss;
    Output = y_ss;
    Investment = i_ss;
    Consumption = c_ss;
    TradeBalance = tb_ss;
    TradeBalanceToOutput = tb_ss / y_ss;
    MUConsumption = mu_c_ss;
    LoggedProductivity = 0;
    TechGrowthRate = g_ss;
    OutputGrowth = g_ss;
    ConsumptionGrowth = g_ss;
    InvestmentGrowth = g_ss;
    PreferenceShock = 1;
    CountryPremiumShock = 1;
end;

shocks;
    var ProductivityShock;
    stderr @{productivity_shock_stderr};
    
    var TechGrowthShock;
    stderr @{tech_growth_shock_stderr};
    
    var PreferenceInnovation;
    stderr @{preference_shock_stderr};
    
    var CountryPremiumInnovation;
    stderr @{country_premium_shock_stderr};
end;

steady;
check;

stoch_simul(order=1, periods=@{periods}, drop=0, irf=0, nomoments, nofunctions, nograph, nocorr, noprint);

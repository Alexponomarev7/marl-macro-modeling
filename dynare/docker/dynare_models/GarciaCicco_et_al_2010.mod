var c           $Consumption$               (long_name='Consumption')
    k           $Capital$                   (long_name='Capital') 
    LoggedProductivity $LoggedProductivity$ (long_name='Total Factor Productivity')
    h           $HoursWorked$               (long_name='Hours Worked')
    d           $Debt$                      (long_name='Debt')
    y           $Output$                    (long_name='Output')
    invest      $Investment$                (long_name='Investment')
    tb          $TradeBalance$              (long_name='Trade Balance')
    mu_c        $MUConsumption$             (long_name='Marginal Utility of Consumption')
    tb_y        ${\frac{TB}{Y}}$            (long_name='Trade Balance to Output Ratio')
    g_y         ${\Delta Y}$                (long_name='Output Growth Rate')
    g_c         ${\Delta C}$                (long_name='Consumption Growth Rate')
    g_invest    ${\Delta I}$                (long_name='Investment Growth Rate')
    g           $TechGrowthRate$            (long_name='Technology Growth Rate')
    r           $InterestRate$              (long_name='Interest Rate')
    mu          $CountryPremiumShock$       (long_name='Country Premium Shock')
    nu          $PreferenceShock$           (long_name='Preference Shock')
;

predetermined_variables k d;

% Define parameters
parameters beta     ${\beta}$     (long_name='Discount Factor')
        gamma_a     ${\gamma_a}$  (long_name='Intertemporal Elasticity of Substitution')
        delta       ${\delta}$    (long_name='Depreciation Rate')
        alpha       ${\alpha}$    (long_name='Capital Elasticity of Production')
        psi         ${\psi}$      (long_name='Debt Elasticity of Interest Rate')
        omega       ${\omega}$    (long_name='Labor Disutility Parameter')
        theta       ${\theta}$    (long_name='Labor Utility Parameter')
        phi         ${\phi}$      (long_name='Capital Adjustment Cost Parameter')
        dbar        ${\bar d}$    (long_name='Steady State Debt')
        gbar        ${\bar g}$    (long_name='Steady State Technology Growth Rate')
        rho_a       ${\rho_a}$    (long_name='Persistence of Temporary Technology Shock')
        rho_g       ${\rho_g}$    (long_name='Persistence of Permanent Technology Shock')
        rho_nu      ${\rho_\nu}$  (long_name='Persistence of Preference Shock')
        rho_mu      ${\rho_\mu}$  (long_name='Persistence of Country Premium Shock')
        rho_s       ${\rho_s}$    (long_name='Persistence of Exogenous Spending Shock')
;

varexo eps_a ${\varepsilon_a}$    (long_name='Temporary Technology Shock')
        eps_g ${\varepsilon_g}$   (long_name='Permanent Technology Shock')
        eps_nu ${\varepsilon_\nu}$ (long_name='Preference Shock')
        eps_mu ${\varepsilon_\mu}$ (long_name='Country Premium Shock')
;

@#if !defined(gbar)
    @#define gbar = 1.0050
@#endif

@#if !defined(rho_g)
    @#define rho_g = 0.8280
@#endif

@#if !defined(rho_a)
    @#define rho_a = 0.7650
@#endif

@#if !defined(phi)
    @#define phi = 3.3000
@#endif

gbar  = @{gbar}; % Gross growth rate of output
rho_g = @{rho_g}; % Serial correlation of innovation in permanent technology shock
rho_a = @{rho_a}; % Serial correlation of transitory technology shock
phi   = @{phi}; % Adjustment cost parameter

% Parameters only used for Financial Frictions model, irrelevant for RBC where volatilities are 0
rho_nu = 0.850328786147732;
rho_s  = 0.205034667802314;
rho_mu = 0.906802888826967;

@#if !defined(gamma_a)
    @#define gamma_a = 2
@#endif

gamma_a = @{gamma_a}; % Intertemporal elasticity of substitution

@#if !defined(delta)
    @#define delta = 1.03^4-1
@#endif

delta = @{delta}; % Depreciation rate

@#if !defined(alpha)
    @#define alpha = 0.32
@#endif

alpha = @{alpha}; % Capital elasticity of the production function

@#if !defined(omega)
    @#define omega = 1.6
@#endif

omega = @{omega}; % Labor disutility parameter

theta = 1.4 * omega; % Labor utility parameter

@#if !defined(beta)
    @#define beta = 0.98^4
@#endif

beta = @{beta}; % Discount factor

@#if !defined(dbar)
    @#define dbar = 0.007
@#endif

dbar = @{dbar}; % Steady state debt

@#if !defined(psi)
    @#define psi = 0.001
@#endif

psi = @{psi};

model;
    % 1. Interest Rate
    r = 1/beta * gbar^gamma_a + psi * (exp(d - dbar) - 1) + exp(mu - 1) - 1;

    % 2. Marginal utility of consumption
    mu_c = nu * (c - theta/omega * h^omega)^(-gamma_a);

    % 3. Resource constraint
    y = log(tb) + c + invest + phi/2 * (k(+1)/k * g - gbar)^2 * k;

    % 4. Trade balance
    log(tb) = d - d(+1) * g / r;

    % 5. Definition output
    y = exp(LoggedProductivity) * k^alpha * (g * h)^(1 - alpha);

    % 6. Definition investment
    invest = k(+1) * g - (1 - delta) * k;

    % 7. Euler equation
    mu_c = beta / g^gamma_a * r * mu_c(+1);

    % 8. First order condition labor
    theta * h^(omega - 1) = (1 - alpha) * exp(LoggedProductivity) * g^(1 - alpha) * (k / h)^alpha;

    % 9. First order condition investment
    mu_c * (1 + phi * (k(+1)/k * g - gbar)) 
        = beta / g^gamma_a * mu_c(+1) 
          * (1 - delta 
             + alpha * exp(LoggedProductivity(+1)) * (g(+1)*h(+1)/k(+1))^(1 - alpha) 
             + phi * k(+2)/k(+1) * g(+1) * (k(+2)/k(+1) * g(+1) - gbar) 
             - phi/2 * (k(+2)/k(+1) * g(+1) - gbar)^2);

    % 10. Definition trade-balance to output ratio
    log(tb_y) = log(tb) / y;

    % 11. Output growth
    g_y = y / y(-1) * g(-1);

    % 12. Consumption growth
    g_c = c / c(-1) * g(-1);

    % 13. Investment growth
    g_invest = invest / invest(-1) * g(-1);

    % 14. LOM temporary TFP
    LoggedProductivity = rho_a * LoggedProductivity(-1) + eps_a;

    % 15. LOM permanent TFP Growth
    log(g / gbar) = rho_g * log(g(-1) / gbar) + eps_g;

    % 16. Preference shock
    log(nu) = rho_nu * log(nu(-1)) + eps_nu;

    % 17. Exogenous stochastic country premium shock
    log(mu) = rho_mu * log(mu(-1)) + eps_mu;
end;

steady_state_model;
    % Вычисляем стационарное состояние
    r = 1 / beta * gbar^gamma_a;
    d = dbar;

    k_over_gh = ((gbar^gamma_a / beta - 1 + delta) / alpha)^(1 / (alpha - 1)); % k / (g*h)
    h = ((1 - alpha) * gbar * k_over_gh^alpha / theta)^(1 / (omega - 1));      % Часы труда
    k = k_over_gh * gbar * h;       % Капитал
    invest = (gbar - 1 + delta) * k; % Инвестиции
    y = k^alpha * (h * gbar)^(1 - alpha); % Выпуск

    s = 0;

    c = (gbar / r - 1) * d + y - s - invest;
    tb = y - c - s - invest;
    tb_y = tb / y;
    mu_c = (c - theta/omega * h^omega)^(-gamma_a);
    LoggedProductivity = 0;
    g = gbar;
    g_c = g;
    g_invest = g;
    g_y = g;
    nu = 1;
    mu = 1;
    tb = exp(tb);
    tb_y = exp(tb_y);
end;

steady;

parameters k_ss c_ss;
k_ss = k;
c_ss = c;

initval;
    k = @{start_capital};
end;

endval;
    k = k_ss;
    c = c_ss;
end;

perfect_foresight_setup(periods = @{periods});
perfect_foresight_solver;

% shocks;
% @#if RBC == 1
%     var eps_a; stderr 0.0270;
%     var eps_g; stderr 0.0300;
%     var eps_nu; stderr 0;
%     var eps_mu; stderr 0;
% @# else
%     var eps_a; stderr 0.033055089525252;
%     var eps_g; stderr 0.010561526060797;
%     var eps_nu; stderr 0.539099453618175;
%     var eps_s; stderr 0.018834174505537;
%     var eps_mu; stderr 0.057195449717680;
% @# endif
% end;
% 
% stoch_simul(order=1, irf=0, periods=50, loglinear, nomoments, nofunctions, nograph, nocorr, noprint);
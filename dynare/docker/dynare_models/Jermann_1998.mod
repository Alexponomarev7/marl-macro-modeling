/*
 * Production economy with internal habits and capital adjustment costs of Urban Jermann (1998):
 * Asset pricing in production economies, Journal of Monetary Economics, 41, pp. 257-275. Adapted
 * from Johannes Pfeifer's DSGE_mod implementation (GPL-3.0), in standard timing (production uses
 * Capital(-1)). Labor is inelastic, so consumption and investment are the decisions.
 */

@#ifndef periods
    @#define periods = 1000
@#endif

var Consumption               $Consumption$               (long_name='Consumption')
    Capital                   $Capital$                   (long_name='Capital')
    Investment                $Investment$                (long_name='Investment')
    LoggedProductivity        $LoggedProductivity$        (long_name='LoggedProductivity')
    MUConsumption             $MUConsumption$             (long_name='MUConsumption')
    TobinsQ                   $TobinsQ$                   (long_name='Tobins marginal q')
    RealWage                  $RealWage$                  (long_name='Real Wage')
    Output                    $Output$                    (long_name='Output')
    RealReturnOnCapital           $RealReturnOnCapital$           (long_name='Real Return On Capital')
    RiskFreeRate              $RiskFreeRate$              (long_name='Risk-Free Rate')
    StochasticDiscountFactor  $StochasticDiscountFactor$  (long_name='stochastic discount factor')
    ConsumptionGrowthRate     $ConsumptionGrowthRate$     (long_name='Consumption Growth Rate')
    OutputGrowthRate          $OutputGrowthRate$          (long_name='Output Growth Rate')
    InvestmentGrowthRate      $InvestmentGrowthRate$      (long_name='Investment Growth Rate')
    HabitAdjustedConsumption  $HabitAdjustedConsumption$  (long_name='habit-adjusted consumption');

varexo ProductivityInnovation $ProductivityInnovation$ (long_name='productivity shock');

parameters betastar  ${\beta^*}$     (long_name='discount factor')
           delta     ${\delta}$      (long_name='depreciation rate')
           alpha     ${\alpha}$      (long_name='capital share')
           rho       ${\rho}$        (long_name='autocorrelation productivity')
           gamma     ${\gamma}$      (long_name='long-run growth rate')
           tau       ${\tau}$        (long_name='risk aversion')
           h         ${h}$           (long_name='habit parameter (alpha in the paper)')
           xi        ${\xi}$         (long_name='elasticity of the investment-capital ratio to q')
           const     ${c}$           (long_name='constant in the investment adjustment costs')
           a         ${a}$           (long_name='parameter a of the investment adjustment costs')
           b         ${b}$           (long_name='parameter b of the investment adjustment costs')
           i_k       ${\frac{I}{K}}$ (long_name='steady-state investment to capital ratio');

% Calibration: Jermann (1998), Table 1 (quarterly)
% (macro name growth_factor: gamma is reserved in the macro language)
@#if !defined(growth_factor)
    @#define growth_factor = 1.005
@#endif
@#if !defined(alpha)
    @#define alpha = 0.36
@#endif
@#if !defined(delta)
    @#define delta = 0.025
@#endif
@#if !defined(tau)
    @#define tau = 5
@#endif
@#if !defined(h)
    @#define h = 0.82
@#endif
@#if !defined(betastar)
    @#define betastar = 0.99393
@#endif
@#if !defined(rho)
    @#define rho = 0.99
@#endif
@#if !defined(xi)
    @#define xi = 0.23
@#endif
% innovation standard deviation (paper: 0.0064/(1-alpha) = 0.01)
@#if !defined(productivity_shock_stderr)
    @#define productivity_shock_stderr = 0.01
@#endif

gamma = @{growth_factor};
alpha = @{alpha};
delta = @{delta};
tau = @{tau};
h = @{h};
betastar = @{betastar};
rho = @{rho};
xi = @{xi};
% adjustment-cost function: normalized so that q = 1 and I/K = i_k in the steady state
a = 1 / xi;
i_k = delta + gamma - 1;
b = i_k^a;
const = i_k - b / (1 - a) * i_k^(1 - a);

model;
[name='Marginal utility (internal habit)']
MUConsumption = (Consumption - h / gamma * Consumption(-1))^(-tau)
    - betastar * h / gamma * (Consumption(+1) - h / gamma * Consumption)^(-tau);
[name='Habit-adjusted consumption']
HabitAdjustedConsumption = Consumption - h / gamma * Consumption(-1);
[name='Real wage']
RealWage = (1 - alpha) * exp(LoggedProductivity) * Capital(-1)^alpha;
[name='Resource constraint']
Consumption + Investment = exp(LoggedProductivity) * Capital(-1)^alpha;
[name='Law of motion of capital']
gamma * Capital = (1 - delta) * Capital(-1) + (b / (1 - a) * (Investment / Capital(-1))^(1 - a) + const) * Capital(-1);
[name='FOC capital']
MUConsumption * TobinsQ * gamma = betastar * MUConsumption(+1) * (alpha * exp(LoggedProductivity(+1)) * Capital^(alpha - 1)
    + TobinsQ(+1) * (1 - delta + const + b * a / (1 - a) * (Investment(+1) / Capital)^(1 - a)));
[name='FOC investment']
1 = TobinsQ * b * (Investment / Capital(-1))^(-a);
[name='Law of motion of technology']
LoggedProductivity = rho * LoggedProductivity(-1) + ProductivityInnovation;
[name='Production function']
Output = exp(LoggedProductivity) * Capital(-1)^alpha;
[name='Return to capital']
RealReturnOnCapital = 1 / TobinsQ * (alpha * exp(LoggedProductivity(+1)) * Capital^(alpha - 1)
    + TobinsQ(+1) * (1 - delta + const + b * a / (1 - a) * (Investment(+1) / Capital)^(1 - a)));
[name='Stochastic discount factor']
StochasticDiscountFactor = betastar / gamma * MUConsumption(+1) / MUConsumption;
[name='Risk-free rate']
RiskFreeRate = 1 / StochasticDiscountFactor;
[name='Growth rate of output']
OutputGrowthRate = log(Output) - log(Output(-1));
[name='Growth rate of investment']
InvestmentGrowthRate = log(Investment) - log(Investment(-1));
[name='Growth rate of consumption']
ConsumptionGrowthRate = log(Consumption) - log(Consumption(-1));
end;

steady_state_model;
    Capital = ((gamma / betastar - (1 - delta + const + b * a / (1 - a) * i_k^(1 - a))) / alpha)^(1 / (alpha - 1));
    TobinsQ = 1;
    LoggedProductivity = 0;
    Investment = i_k * Capital;
    RealWage = (1 - alpha) * Capital^alpha;
    Output = Capital^alpha;
    Consumption = Output - Investment;
    HabitAdjustedConsumption = Consumption * (1 - h / gamma);
    MUConsumption = (Consumption * (1 - h / gamma))^(-tau) * (1 - betastar * h / gamma);
    RealReturnOnCapital = alpha * Capital^(alpha - 1) + TobinsQ * (1 - delta + const + b * a / (1 - a) * i_k^(1 - a));
    StochasticDiscountFactor = betastar / gamma;
    RiskFreeRate = gamma / betastar;
    OutputGrowthRate = 0;
    InvestmentGrowthRate = 0;
    ConsumptionGrowthRate = 0;
end;

shocks;
    var ProductivityInnovation = @{productivity_shock_stderr}^2;
end;

steady;
check;

stoch_simul(order=1, periods=@{periods}, irf=0, nomoments, nofunctions, nograph, nocorr, noprint);

/*
 * Jermann, Urban and Vincenzo Quadrini (2012): "Macroeconomic effects of financial shocks",
 * American Economic Review 102(1): 238-271, RBC version (Section II). Adapted from Johannes
 * Pfeifer's DSGE_mod implementation (GPL-3.0), with an analytic steady state (xi_ss calibrated).
 * The agent is the firm: investment, new debt and dividends under the enforcement constraint.
 */

@#ifndef periods
    @#define periods = 1000
@#endif

var Consumption                 $Consumption$                 (long_name='Consumption')
    Labor                       $Labor$                       (long_name='Labor')
    Wage                        $Wage$                        (long_name='Wage')
    Capital                     $Capital$                     (long_name='Capital')
    EffectiveGrossInterestRate  $EffectiveGrossInterestRate$  (long_name='Effective Gross Interest Rate')
    InterestRate                $InterestRate$                (long_name='Interest Rate')
    Dividends                   $Dividends$                   (long_name='Dividends')
    Debt                        $Debt$                        (long_name='Debt')
    EnforcementMultiplier       $EnforcementMultiplier$       (long_name='Enforcement Constraint Multiplier')
    FirmValue                   $FirmValue$                   (long_name='Firm Value')
    Productivity                $Productivity$                (long_name='Productivity')
    FinancialConditions         $FinancialConditions$         (long_name='Financial Conditions')
    Output                      $Output$                      (long_name='Output')
    Investment                  $Investment$                  (long_name='Investment');

varexo ProductivityInnovation  $ProductivityInnovation$  (long_name='Technology shock')
       FinancialInnovation     $FinancialInnovation$     (long_name='Financial shock');

parameters theta    ${\theta}$     (long_name='capital share')
           betta    ${\beta}$      (long_name='discount factor')
           alppha   ${\alpha}$     (long_name='disutility from work')
           delta    ${\delta}$     (long_name='depreciation')
           tau      ${\tau}$       (long_name='tax wedge (debt tax advantage)')
           kappa    ${\kappa}$     (long_name='equity payout adjustment cost')
           siggma   ${\sigma}$     (long_name='risk aversion')
           xi_ss    ${\bar\xi}$    (long_name='steady-state enforcement tightness')
           n_target ${\bar n}$     (long_name='steady-state hours')
           A11      ${A_{11}}$     (long_name='VAR coefficient z on z(-1)')
           A12      ${A_{12}}$     (long_name='VAR coefficient z on xi(-1)')
           A21      ${A_{21}}$     (long_name='VAR coefficient xi on z(-1)')
           A22      ${A_{22}}$     (long_name='VAR coefficient xi on xi(-1)');

% Calibration: Jermann & Quadrini (2012), Table 1 and Section III (replication values)
@#if !defined(theta)
    @#define theta = 0.36
@#endif
@#if !defined(betta)
    @#define betta = 0.9825
@#endif
@#if !defined(delta)
    @#define delta = 0.025
@#endif
@#if !defined(tau)
    @#define tau = 0.35
@#endif
@#if !defined(kappa)
    @#define kappa = 0.146
@#endif
@#if !defined(siggma)
    @#define siggma = 1
@#endif
@#if !defined(xi_ss)
    @#define xi_ss = 0.16337753
@#endif
@#if !defined(n_target)
    @#define n_target = 0.3
@#endif
@#if !defined(A11)
    @#define A11 = 0.9457
@#endif
@#if !defined(A12)
    @#define A12 = -0.0091
@#endif
@#if !defined(A21)
    @#define A21 = 0.0321
@#endif
@#if !defined(A22)
    @#define A22 = 0.9703
@#endif
@#if !defined(productivity_shock_stderr)
    @#define productivity_shock_stderr = 0.0045
@#endif
@#if !defined(financial_shock_stderr)
    @#define financial_shock_stderr = 0.0098
@#endif

theta = @{theta};
betta = @{betta};
delta = @{delta};
tau = @{tau};
kappa = @{kappa};
siggma = @{siggma};
xi_ss = @{xi_ss};
n_target = @{n_target};
A11 = @{A11};
A12 = @{A12};
A21 = @{A21};
A22 = @{A22};

model;
#d_ss = steady_state(Dividends);
#payout_cost = 1 + 2 * kappa * (Dividends - d_ss);
#payout_cost_next = 1 + 2 * kappa * (Dividends(+1) - d_ss);
[name='FOC labor (household)']
Wage / Consumption^siggma - alppha / (1 - Labor) = 0;
[name='Euler equation (household)']
Consumption^(-siggma) = betta * ((EffectiveGrossInterestRate - tau) / (1 - tau)) * Consumption(+1)^(-siggma);
[name='Budget constraint household']
Wage * Labor + Debt(-1) - Debt / EffectiveGrossInterestRate + Dividends - Consumption = 0;
[name='FOC labor input (firm)']
(1 - theta) * Productivity * Capital(-1)^theta * Labor^(-theta) = Wage * (1 / (1 - EnforcementMultiplier * payout_cost));
[name='FOC capital (firm)']
betta * (Consumption / Consumption(+1))^siggma * (payout_cost / payout_cost_next) * (1 - delta + (1 - EnforcementMultiplier(+1) * payout_cost_next) * theta * Productivity(+1) * Capital^(theta - 1) * Labor(+1)^(1 - theta)) + FinancialConditions * EnforcementMultiplier * payout_cost = 1;
[name='FOC bonds (firm)']
EffectiveGrossInterestRate * betta * (Consumption / Consumption(+1))^siggma * (payout_cost / payout_cost_next) + FinancialConditions * EnforcementMultiplier * payout_cost * (EffectiveGrossInterestRate * (1 - tau) / (EffectiveGrossInterestRate - tau)) = 1;
[name='Budget constraint firm']
(1 - delta) * Capital(-1) + Productivity * Capital(-1)^theta * Labor^(1 - theta) - Wage * Labor - Debt(-1) + Debt / EffectiveGrossInterestRate - Capital - (Dividends + kappa * (Dividends - d_ss)^2) = 0;
[name='Enforcement constraint']
FinancialConditions * (Capital - Debt * ((1 - tau) / (EffectiveGrossInterestRate - tau))) = Productivity * Capital(-1)^theta * Labor^(1 - theta);
[name='VAR for productivity, eq. (11)']
log(Productivity) = A11 * log(Productivity(-1)) + A12 * log(FinancialConditions(-1) / xi_ss) + ProductivityInnovation;
[name='VAR for financial conditions, eq. (11)']
log(FinancialConditions / xi_ss) = A21 * log(Productivity(-1)) + A22 * log(FinancialConditions(-1) / xi_ss) + FinancialInnovation;
[name='Production function']
Output = Productivity * Capital(-1)^theta * Labor^(1 - theta);
[name='Law of motion capital']
Investment = Capital - (1 - delta) * Capital(-1);
[name='Firm value']
FirmValue = Dividends + betta * (Consumption / Consumption(+1))^siggma * FirmValue(+1);
[name='Before-tax net interest rate']
InterestRate = (EffectiveGrossInterestRate - tau) / (1 - tau) - 1;
end;

steady_state_model;
Productivity = 1;
FinancialConditions = xi_ss;
Labor = n_target;
EffectiveGrossInterestRate = (1 - tau) / betta + tau;
EnforcementMultiplier = (1 - EffectiveGrossInterestRate * betta) / (xi_ss * (EffectiveGrossInterestRate * (1 - tau) / (EffectiveGrossInterestRate - tau)));
Capital = ((((1 - xi_ss * EnforcementMultiplier) / betta) - (1 - delta)) / ((1 - EnforcementMultiplier) * theta * Labor^(1 - theta)))^(1 / (theta - 1));
Wage = (1 - theta) * Capital^theta * Labor^(-theta) * (1 - EnforcementMultiplier);
Debt = (Capital^theta * Labor^(1 - theta) / xi_ss - Capital) * (tau - EffectiveGrossInterestRate) / (1 - tau);
Dividends = (1 - delta) * Capital + Capital^theta * Labor^(1 - theta) - Wage * Labor - Debt + Debt / EffectiveGrossInterestRate - Capital;
Consumption = Wage * Labor + Debt - Debt / EffectiveGrossInterestRate + Dividends;
alppha = Wage / Consumption^siggma * (1 - Labor);
Output = Capital^theta * Labor^(1 - theta);
Investment = delta * Capital;
FirmValue = Dividends / (1 - betta);
InterestRate = (EffectiveGrossInterestRate - tau) / (1 - tau) - 1;
end;

shocks;
    var ProductivityInnovation; stderr @{productivity_shock_stderr};
    var FinancialInnovation; stderr @{financial_shock_stderr};
end;

steady;
check;

stoch_simul(order=1, periods=@{periods}, irf=0, nomoments, nofunctions, nograph, nocorr, noprint);

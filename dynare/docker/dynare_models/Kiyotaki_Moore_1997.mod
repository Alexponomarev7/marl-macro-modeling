/*
 * Credit-cycle model of Kiyotaki and Moore (1997): Credit Cycles, Journal of Political Economy,
 * 105(2), pp. 211-248, Section 2, in the version of Eric Sims' 2020 lecture notes. Adapted from
 * Johannes Pfeifer's DSGE_mod implementation (GPL-3.0). Impatient farmers hold land and borrow
 * against it (Debt = betap * E[LandPrice(+1)] * Land); patient gatherers lend. Utility is linear.
 */

@#ifndef periods
    @#define periods = 1000
@#endif

var FarmerConsumption      $FarmerConsumption$      (long_name='consumption farmer')
    GathererConsumption    $GathererConsumption$    (long_name='consumption gatherer')
    Debt                   $Debt$                   (long_name='Debt')
    Land                   $Land$                   (long_name='Land')
    GathererLand           $GathererLand$           (long_name='land held by gatherers')
    LandPrice              $LandPrice$              (long_name='LandPrice')
    EnforcementMultiplier  $EnforcementMultiplier$  (long_name='EnforcementMultiplier')
    ConsumptionMultiplier  $ConsumptionMultiplier$  (long_name='multiplier on the farmer consumption constraint')
    Consumption            $Consumption$            (long_name='Consumption')
    Output                 $Output$                 (long_name='Output')
    TechnologyShock        $TechnologyShock$        (long_name='Technology Shock');

varexo ed $ed$ (long_name='productivity shock');

parameters alpha  $\alpha$           (long_name='exponent production function gatherers')
           m      $m$                (long_name='population size gatherers')
           K_bar  $\bar K$           (long_name='land supply')
           beta   $\beta$            (long_name='discount factor farmer')
           betap  ${\beta^\prime}$   (long_name='discount factor gatherer')
           a      $a$                (long_name='tradable share of fruit')
           c      $c$                (long_name='non-tradable share of fruit')
           z      $z$                (long_name='constant production function gatherers');

% Calibration: Sims (2020) lecture notes
@#if !defined(alpha)
    @#define alpha = 1/3
@#endif
@#if !defined(m)
    @#define m = 0.5
@#endif
@#if !defined(beta)
    @#define beta = 0.98
@#endif
@#if !defined(betap)
    @#define betap = 0.99
@#endif
@#if !defined(a)
    @#define a = 0.7
@#endif
@#if !defined(c)
    @#define c = 0.3
@#endif
@#if !defined(z)
    @#define z = 0.01
@#endif
@#if !defined(productivity_shock_stderr)
    @#define productivity_shock_stderr = 0.0011
@#endif

alpha = @{alpha};
m = @{m};
K_bar = 1;
beta = @{beta};
betap = @{betap};
a = @{a};
c = @{c};
z = @{z};

model;
[name='Euler equation bonds farmer']
1 + ConsumptionMultiplier = (beta * (1 + ConsumptionMultiplier(+1)) + EnforcementMultiplier) / betap;
[name='Euler equation land farmer']
LandPrice * (1 + ConsumptionMultiplier) + beta * c * ConsumptionMultiplier(+1)
    = beta * (1 + ConsumptionMultiplier(+1)) * ((1 + ed(+1)) * (a + c) + LandPrice(+1)) + EnforcementMultiplier * LandPrice(+1);
[name='Budget constraint farmer']
LandPrice * (Land - Land(-1)) + Debt(-1) / betap + FarmerConsumption = (1 + ed) * (a + c) * Land(-1) + Debt;
[name='Borrowing constraint']
Debt = betap * LandPrice(+1) * Land;
[name='Euler equation gatherer']
LandPrice = betap * ((1 + ed(+1)) * alpha * (z + GathererLand)^(alpha - 1) + LandPrice(+1));
[name='Resource constraint']
FarmerConsumption + m * GathererConsumption = (1 + ed) * (a + c) * Land(-1) + m * (1 + ed) * (z + GathererLand(-1))^alpha;
[name='Land market clearing']
Land + m * GathererLand = K_bar;
[name='Non-tradable constraint']
FarmerConsumption = c * Land(-1);
[name='Aggregate consumption']
Consumption = FarmerConsumption + m * GathererConsumption;
[name='Aggregate output']
Output = Consumption;
[name='Current productivity shock']
TechnologyShock = ed;
end;

steady_state_model;
    LandPrice = a / (1 - betap);
    GathererLand = (betap * alpha / a)^(1 / (1 - alpha)) - z;
    Land = K_bar - m * GathererLand;
    Debt = betap * LandPrice * Land;
    GathererConsumption = (1 / m) * (a * Land + m * (z + GathererLand)^alpha);
    ConsumptionMultiplier = (a * (beta - 1) + beta * c) / (a * (1 - beta));
    EnforcementMultiplier = (betap - beta) * beta * c / (a * (1 - beta));
    FarmerConsumption = c * Land;
    Consumption = FarmerConsumption + m * GathererConsumption;
    Output = Consumption;
    TechnologyShock = 0;
end;

shocks;
    var ed = @{productivity_shock_stderr}^2;
end;

steady;
check;

stoch_simul(order=1, periods=@{periods}, irf=0, nomoments, nofunctions, nograph, nocorr, noprint);

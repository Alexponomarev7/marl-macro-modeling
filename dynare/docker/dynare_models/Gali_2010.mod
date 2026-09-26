/*
 * Sticky-wage New Keynesian model with unemployment of Jordi Gali (2010): Monetary Policy and
 * Unemployment, Handbook of Monetary Economics, Volume 3A, Chapter 10. Adapted from the DSGE_mod
 * implementation of Lahcen Bounader and Johannes Pfeifer (GPL-3.0); the steady-state calibration
 * identities are parameter expressions. Log deviations from the steady state, in percent.
 */

@#ifndef periods
    @#define periods = 1000
@#endif

var OutputDeviationFromSteadyState  $OutputDeviationFromSteadyState$  (long_name='Output Deviation From Steady State')
    Consumption          $Consumption$          (long_name='Consumption')
    RealInterestRate     $RealInterestRate$     (long_name='Real Interest Rate')
    NominalInterestRate  $NominalInterestRate$  (long_name='Nominal Interest Rate')
    Employment           $Employment$           (long_name='Employment')
    LaborEffort          $LaborEffort$          (long_name='labor effort')
    LaborForce           $LaborForce$           (long_name='labor force')
    Unemployment         $Unemployment$         (long_name='unemployment')
    UnemploymentStart    $UnemploymentStart$    (long_name='unemployment at the beginning of the period')
    UnemploymentRate     $UnemploymentRate$     (long_name='Unemployment Rate')
    JobFindingRate       $JobFindingRate$       (long_name='job finding rate')
    HiringCost           $HiringCost$           (long_name='cost of hiring')
    Hires                $Hires$                (long_name='new hires')
    Markup               $Markup$               (long_name='Markup')
    RealWage             $RealWage$             (long_name='Real Wage')
    HiringCostComposite  $HiringCostComposite$  (long_name='composite auxiliary variable')
    TargetWage           $TargetWage$           (long_name='targeted wage (Nash bargaining without rigidities)')
    TechnologyShock      $TechnologyShock$      (long_name='Technology Shock')
    WageInflation        $WageInflation$        (long_name='Wage Inflation')
    PriceInflation       $PriceInflation$       (long_name='Price Inflation')
    MonetaryShock        $MonetaryShock$        (long_name='Monetary Policy Shock');

varexo TechnologyInnovation  $TechnologyInnovation$  (long_name='Technology Shock')
       MonetaryInnovation    $MonetaryInnovation$    (long_name='Monetary Policy Shock');

parameters alfa     ${\alpha}$      (long_name='exponent of labor in the production function')
           delta    ${\delta}$      (long_name='separation rate')
           gammma   ${\gamma}$      (long_name='elasticity of the hiring cost to labor market tightness')
           psi      ${\psi}$        (long_name='weight of unemployment in labor market effort')
           betta    ${\beta}$       (long_name='discount factor')
           rho_a    ${\rho_a}$      (long_name='autocorrelation technology shock')
           rho_nu   ${\rho_\nu}$    (long_name='autocorrelation monetary policy shock')
           varphi   ${\varphi}$     (long_name='inverse Frisch elasticity of labor effort')
           xi       ${\xi}$         (long_name='bargaining power of workers')
           theta_w  ${\theta_w}$    (long_name='Calvo wage rigidity')
           theta_p  ${\theta_p}$    (long_name='Calvo price rigidity')
           phi_pi   ${\phi_{\pi}}$  (long_name='Taylor rule inflation coefficient')
           phi_y    ${\phi_y}$      (long_name='Taylor rule output coefficient')
           Gamma    ${\Gamma}$      (long_name='scale of the hiring cost')
           Theta    ${\Theta}$      (long_name='hiring costs to GDP')
           Upsilon  ${\Upsilon}$    (long_name='MRS weight in the target wage')
           Phi      ${\Phi}$        (long_name='hiring costs share of hiring costs plus wage')
           Xi       ${\Xi}$         (long_name='wage-inflation coefficient in the participation condition')
           chi      ${\chi}$        (long_name='labor disutility')
           N        ${N}$           (long_name='employment rate')
           U        ${U}$           (long_name='unemployment rate')
           F        ${F}$           (long_name='labor force')
           L        ${L}$           (long_name='labor market effort')
           x        ${x}$           (long_name='job finding rate')
           G_hc     ${G}$           (long_name='hiring cost per hire')
           B_hc     ${B}$           (long_name='hiring cost term of the hiring condition')
           MRPN     ${MRPN}$        (long_name='marginal revenue product of labor')
           C_ss     ${C}$           (long_name='consumption')
           MRS      ${MRS}$         (long_name='marginal rate of substitution')
           W_div_P  ${W/P}$         (long_name='real wage');

% Calibration: Gali (2010), pp. 515-517 (quarterly)
@#if !defined(N)
    @#define N = 0.59
@#endif
@#if !defined(U)
    @#define U = 0.03
@#endif
@#if !defined(x)
    @#define x = 0.7
@#endif
@#if !defined(alfa)
    @#define alfa = 1/3
@#endif
@#if !defined(betta)
    @#define betta = 0.99
@#endif
@#if !defined(varphi)
    @#define varphi = 5
@#endif
@#if !defined(theta_w)
    @#define theta_w = 0.75
@#endif
@#if !defined(theta_p)
    @#define theta_p = 0.75
@#endif
@#if !defined(gammma)
    @#define gammma = 1
@#endif
@#if !defined(Theta)
    @#define Theta = 0.0014
@#endif
@#if !defined(xi)
    @#define xi = 0.05
@#endif
@#if !defined(phi_pi)
    @#define phi_pi = 1.5
@#endif
@#if !defined(phi_y)
    @#define phi_y = 0.125
@#endif
@#if !defined(rho_a)
    @#define rho_a = 0.9
@#endif
@#if !defined(rho_nu)
    @#define rho_nu = 0.5
@#endif
@#if !defined(technology_shock_stderr)
    @#define technology_shock_stderr = 1
@#endif
@#if !defined(monetary_shock_stderr)
    @#define monetary_shock_stderr = 0.25
@#endif

N = @{N};
U = @{U};
x = @{x};
alfa = @{alfa};
betta = @{betta};
varphi = @{varphi};
theta_w = @{theta_w};
theta_p = @{theta_p};
gammma = @{gammma};
Theta = @{Theta};
xi = @{xi};
phi_pi = @{phi_pi};
phi_y = @{phi_y};
rho_a = @{rho_a};
rho_nu = @{rho_nu};

% Calibration identities (Gali_2010_steadystate.m)
F = N + U;
delta = x / (1 - x) * U / N;
Gamma = Theta / (N^alfa * x^gammma * delta);
G_hc = Gamma * x^gammma;
B_hc = (1 - betta * (1 - delta)) * G_hc;
MRPN = (1 - alfa) * N^(-alfa);
C_ss = N^(1 - alfa) - delta * N * G_hc;
psi = (1 - xi) * G_hc * x / ((1 - x) * (xi * MRPN - B_hc));
L = N + psi * U;
chi = (1 - xi) * G_hc * (x / (1 - x)) / (xi * psi * C_ss * L^varphi);
MRS = chi * C_ss * L^varphi;
W_div_P = xi * MRS + (1 - xi) * MRPN;
Upsilon = xi * MRS / W_div_P;
Phi = B_hc / (W_div_P + B_hc);
Xi = (xi * W_div_P / (G_hc * (1 - xi))) * (theta_w / ((1 - theta_w) * (1 - betta * theta_w * (1 - delta))));

model(linear);
#lambda_p = (1 - theta_p) * (1 - betta * theta_p) / theta_p;
#lambda_w = (1 - betta * (1 - delta) * theta_w) * (1 - theta_w) / (theta_w * (1 - (1 - Upsilon) * (1 - Phi)));

[name='1. Goods market clearing']
OutputDeviationFromSteadyState = (1 - Theta) * Consumption + Theta * (HiringCost + Hires);
[name='2. Aggregate production function']
OutputDeviationFromSteadyState = TechnologyShock + (1 - alfa) * Employment;
[name='3. Aggregate hiring and employment']
delta * Hires = Employment - (1 - delta) * Employment(-1);
[name='4. Hiring cost']
HiringCost = gammma * JobFindingRate;
[name='5. Job finding rate']
JobFindingRate = Hires - UnemploymentStart;
[name='6. Effective market effort']
LaborEffort = (N / L) * Employment + (psi * U / L) * Unemployment;
[name='7. Labor force']
LaborForce = (N / F) * Employment + (U / F) * Unemployment;
[name='8. Unemployment']
Unemployment = UnemploymentStart - (x / (1 - x)) * JobFindingRate;
[name='9. Unemployment rate (percentage points)']
UnemploymentRate = U / F * Unemployment - U / F * LaborForce;
[name='10. Euler equation']
Consumption = Consumption(+1) - RealInterestRate;
[name='11. Fisher equation']
RealInterestRate = NominalInterestRate - PriceInflation(+1);
[name='12. Price Phillips curve']
PriceInflation = betta * PriceInflation(+1) - lambda_p * Markup;
[name='13. Optimal hiring condition']
alfa * Employment = TechnologyShock - ((1 - Phi) * RealWage + Phi * HiringCostComposite) - Markup;
[name='14. Definition of the hiring-cost composite']
HiringCostComposite = (1 / (1 - betta * (1 - delta))) * HiringCost
    - (betta * (1 - delta) / (1 - betta * (1 - delta))) * (HiringCost(+1) - RealInterestRate);
[name='15. Optimal participation condition']
Consumption + varphi * LaborEffort = (1 / (1 - x)) * JobFindingRate + HiringCost - Xi * WageInflation;
[name='16. Evolution of the real wage']
RealWage = RealWage(-1) + WageInflation - PriceInflation;
[name='17. Wage Phillips curve']
WageInflation = betta * (1 - delta) * WageInflation(+1) - lambda_w * (RealWage - TargetWage);
[name='18. Target wage']
TargetWage = Upsilon * (Consumption + varphi * LaborEffort) + (1 - Upsilon) * (-Markup + TechnologyShock - alfa * Employment);
[name='19. Interest rate rule']
NominalInterestRate = phi_pi * PriceInflation + phi_y * OutputDeviationFromSteadyState + MonetaryShock;
[name='20. Monetary policy shock']
MonetaryShock = rho_nu * MonetaryShock(-1) + MonetaryInnovation;
[name='21. Technology']
TechnologyShock = rho_a * TechnologyShock(-1) + TechnologyInnovation;
end;

shocks;
    var TechnologyInnovation = @{technology_shock_stderr}^2;
    var MonetaryInnovation = @{monetary_shock_stderr}^2;
end;

steady;
check;

stoch_simul(order=1, periods=@{periods}, irf=0, nomoments, nofunctions, nograph, nocorr, noprint);

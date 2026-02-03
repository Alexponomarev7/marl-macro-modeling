% Faia (2008): Optimal Monetary Policy Rules with Labor Market Frictions
%
% Key features:
%   - Closed economy New Keynesian model
%   - Search and matching labor market (Blanchard-Diamond)
%   - Rotemberg price adjustment costs
%   - Nash wage bargaining with wage rigidity
%   - Taylor rule monetary policy

var LagrangeMultiplierA     $LagrangeMultiplierA$     (long_name='marginal utility of consumption')
    Consumption             $Consumption$             (long_name='consumption')
    NominalInterestRate     $NominalInterestRate$     (long_name='nominal interest rate')
    InflationRate           $InflationRate$           (long_name='gross inflation rate')
    MarketTightness         $MarketTightness$         (long_name='labor market tightness')
    Vacancies               $Vacancies$               (long_name='vacancies')
    UnemploymentRate        $UnemploymentRate$        (long_name='unemployment rate')
    Matches                 $Matches$                 (long_name='matches')
    MeetingRate             $MeetingRate$             (long_name='meeting rate firms and workers')
    Employment              $Employment$              (long_name='employment')
    GrossOutputA            $GrossOutputA$            (long_name='gross output before costs')
    GrossOutputB            $GrossOutputB$            (long_name='net output after costs')
    LagrangeMultiplierB     $LagrangeMultiplierB$     (long_name='marginal value of employment')
    LogTFP                  $LogTFP$                  (long_name='log total factor productivity')
    MarginalCosts           $MarginalCosts$           (long_name='real marginal cost')
    RealWage                $RealWage$                (long_name='real wage')
    GovernmentSpending      $GovernmentSpending$      (long_name='government spending')
    GovernmentSpendingShock $GovernmentSpendingShock$ (long_name='government spending shock process')
    LogOutput               $LogOutput$               (long_name='log net output')
    LogVacancies            $LogVacancies$            (long_name='log vacancies')
    LogWages                $LogWages$                (long_name='log real wage')
    LogUnemployment         $LogUnemployment$         (long_name='log unemployment')
    LogTightnessA           $LogTightnessA$           (long_name='log market tightness')
    LogTightnessB           $LogTightnessB$           (long_name='log inflation');

varexo GovernmentSpendingInnovation   $GovernmentSpendingInnovation$   (long_name='government spending innovation')
       ProductivityInnovation         $ProductivityInnovation$         (long_name='TFP innovation');

 
parameters epsilon         ${\varepsilon}$   (long_name='substitution elasticity')
           psi_p           ${\psi}$          (long_name='price adjustment costs')
           beta            ${\beta}$         (long_name='discount factor')
           xi              ${\xi}$           (long_name='matching function exponent')
           varsigma        ${\varsigma}$     (long_name='worker bargaining power')
           rho_sep         ${\rho}$          (long_name='separation rate')
           m_scale         ${m}$             (long_name='matching efficiency')
           b_w_target      ${b/w}$           (long_name='replacement ratio target')
           b_unemp         ${b}$             (long_name='unemployment benefits')
           kappa           ${\kappa}$        (long_name='vacancy posting cost')
           lambda_w        ${\lambda}$       (long_name='wage rigidity')
           g_share         ${g/y}$           (long_name='government spending share')
           g_ss            ${\bar{G}}$       (long_name='steady state government spending')
           rho_g           ${\rho_G}$        (long_name='government spending persistence')
           rho_z           ${\rho_Z}$        (long_name='TFP persistence')
           sigma           ${\sigma}$        (long_name='risk aversion')
           phi_r           ${\phi_R}$        (long_name='interest rate smoothing')
           phi_pi          ${\phi_\pi}$      (long_name='inflation response')
           phi_y           ${\phi_y}$        (long_name='output response')
           phi_u           ${\phi_u}$        (long_name='unemployment response')
           NominalInterestRate_ss MarginalCosts_ss MeetingRate_ss MarketTightness_ss
           UnemploymentRate_ss Employment_ss Vacancies_ss Matches_ss RealWage_ss
           LagrangeMultiplierB_ss GrossOutputA_ss GrossOutputB_ss Consumption_ss
           LagrangeMultiplierA_ss;

beta = @{beta};
sigma = @{sigma};
epsilon = @{epsilon};
psi_p = @{psi_p};
xi = @{xi};
rho_sep = @{rho_sep};
varsigma = @{varsigma};
b_w_target = @{b_w_target};
lambda_w = @{lambda_w};
g_share = @{g_share};
rho_z = @{rho_z};
rho_g = @{rho_g};

phi_r = 0.9;
phi_pi = 5;
phi_y = 0;
phi_u = 0;

NominalInterestRate_ss = 1 / beta;
MarginalCosts_ss = (epsilon - 1) / epsilon;
MeetingRate_ss = 0.7;
MarketTightness_ss = 0.6 / MeetingRate_ss;
m_scale = MeetingRate_ss * MarketTightness_ss^xi;
UnemploymentRate_ss = 1 / (1 + MeetingRate_ss * MarketTightness_ss * (1 - rho_sep) / rho_sep);
Employment_ss = 1 - UnemploymentRate_ss;
Vacancies_ss = rho_sep * Employment_ss / ((1 - rho_sep) * MeetingRate_ss);
Matches_ss = m_scale * (UnemploymentRate_ss^xi) * (Vacancies_ss^(1 - xi));
RealWage_ss = ((1 - (1 - varsigma) * b_w_target) / (MeetingRate_ss * beta * (1 - rho_sep) * varsigma * MarketTightness_ss) + 1 / (1 - beta * (1 - rho_sep)))^(-1) * MarginalCosts_ss * (varsigma / (MeetingRate_ss * beta * (1 - rho_sep) * varsigma * MarketTightness_ss) + 1 / (1 - beta * (1 - rho_sep)));
LagrangeMultiplierB_ss = (MarginalCosts_ss - RealWage_ss) / (1 - beta * (1 - rho_sep));
kappa = LagrangeMultiplierB_ss * (MeetingRate_ss * beta * (1 - rho_sep));
GrossOutputA_ss = Employment_ss;
GrossOutputB_ss = GrossOutputA_ss - kappa * Vacancies_ss;
g_ss = g_share * GrossOutputB_ss;
Consumption_ss = GrossOutputB_ss - g_ss;
LagrangeMultiplierA_ss = Consumption_ss^(-sigma);
b_unemp = b_w_target * RealWage_ss;

model;

[name='Marginal utility, eq. (3)']
LagrangeMultiplierA = Consumption^(-sigma);

[name='Euler equation, eq. (4)']
1 / NominalInterestRate = beta * (LagrangeMultiplierA(+1) / LagrangeMultiplierA) / InflationRate(+1);

[name='Labor market tightness']
MarketTightness = Vacancies / UnemploymentRate;

[name='Matching function, eq. (5)']
Matches = m_scale * (UnemploymentRate^xi) * (Vacancies^(1 - xi));

[name='Meeting rate between firms and workers']
MeetingRate = Matches / Vacancies;

[name='Production function, eq. (6)']
GrossOutputA = exp(LogTFP) * Employment;

[name='TFP process']
LogTFP = rho_z * LogTFP(-1) + ProductivityInnovation;

[name='Employment dynamics, eq. (7)']
Employment = (1 - rho_sep) * (Employment(-1) + Vacancies(-1) * MeetingRate(-1));

[name='Unemployment identity, eq. (8)']
UnemploymentRate = 1 - Employment;

[name='FOC labor input, eq. (13)']
LagrangeMultiplierB = MarginalCosts * exp(LogTFP) - RealWage + beta * (LagrangeMultiplierA(+1) / LagrangeMultiplierA) * (1 - rho_sep) * LagrangeMultiplierB(+1);

[name='FOC vacancy posting, eq. (14)']
kappa / MeetingRate = beta * (LagrangeMultiplierA(+1) / LagrangeMultiplierA) * (1 - rho_sep) * LagrangeMultiplierB(+1);

[name='Phillips curve (Rotemberg), eq. (15)']
1 - psi_p * (InflationRate - 1) * InflationRate + beta * (LagrangeMultiplierA(+1) / LagrangeMultiplierA) * (psi_p * (InflationRate(+1) - 1) * InflationRate(+1) * GrossOutputA(+1) / GrossOutputA) = (1 - MarginalCosts) * epsilon;

[name='Nash wage, eq. (24)']
RealWage = lambda_w * (varsigma * (MarginalCosts * exp(LogTFP) + MarketTightness * kappa) + (1 - varsigma) * b_unemp) + (1 - lambda_w) * steady_state(RealWage);

[name='Market clearing']
GrossOutputB = Consumption + GovernmentSpending;

[name='Net output after adjustment costs']
GrossOutputB = GrossOutputA - kappa * Vacancies - GrossOutputA * (psi_p / 2) * (InflationRate - 1)^2;

[name='Taylor rule, eq. (25)']
log(NominalInterestRate / steady_state(NominalInterestRate)) = phi_r * log(NominalInterestRate(-1) / steady_state(NominalInterestRate)) + (1 - phi_r) * (phi_pi * log(InflationRate) + phi_y * log(GrossOutputB / GrossOutputB(-1)) + phi_u * log(UnemploymentRate / steady_state(UnemploymentRate)));

[name='Government spending level']
GovernmentSpending = g_ss * exp(GovernmentSpendingShock);

[name='Government spending shock process']
GovernmentSpendingShock = rho_g * GovernmentSpendingShock(-1) + GovernmentSpendingInnovation;

[name='Log output']
LogOutput = log(GrossOutputB);

[name='Log vacancies']
LogVacancies = log(Vacancies);

[name='Log wages']
LogWages = log(RealWage);

[name='Log unemployment']
LogUnemployment = log(UnemploymentRate);

[name='Log market tightness']
LogTightnessA = log(MarketTightness);

[name='Log inflation']
LogTightnessB = log(InflationRate);

end;

steady_state_model;
    LogTFP = 0;
    GovernmentSpendingShock = 0;
    InflationRate = 1;
    NominalInterestRate = NominalInterestRate_ss;
    MarginalCosts = MarginalCosts_ss;
    MeetingRate = MeetingRate_ss;
    MarketTightness = MarketTightness_ss;
    UnemploymentRate = UnemploymentRate_ss;
    Employment = Employment_ss;
    Vacancies = Vacancies_ss;
    Matches = Matches_ss;
    RealWage = RealWage_ss;
    LagrangeMultiplierB = LagrangeMultiplierB_ss;
    GrossOutputA = GrossOutputA_ss;
    GrossOutputB = GrossOutputB_ss;
    GovernmentSpending = g_ss;
    Consumption = Consumption_ss;
    LagrangeMultiplierA = LagrangeMultiplierA_ss;
    LogOutput = log(GrossOutputB);
    LogVacancies = log(Vacancies);
    LogWages = log(RealWage);
    LogUnemployment = log(UnemploymentRate);
    LogTightnessA = log(MarketTightness);
    LogTightnessB = log(InflationRate);
end;

steady;
check;

shocks;
    var ProductivityInnovation;
    stderr @{stderr_z};
    var GovernmentSpendingInnovation;
    stderr @{stderr_g};
end;

stoch_simul(order=1, periods=@{periods}, drop=0, irf=0, nomoments, nofunctions, nograph, nocorr, noprint);

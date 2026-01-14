% Important note: aponomarev
% At i-th row Capital is the result of the consumption at i-th row
% So the actual state is K(-1) Y

% Model features:
% - Cobb-Douglas production: Y = K^alpha * (A * L)^(1-alpha)
% - Exogenous labor/population growth: L_t = (1+n) * L_{t-1}
% - Exogenous technological progress: A_t = (1+g) * A_{t-1}
% - Variables per effective labor: k = K/(A * L), c = C/(A * L), y = Y/(A * L)
% - CRRA utility function: U = c ^ (1 - sigma) / (1 - sigma)
% - Interest rate: r = dY/dK - delta = MPK - delta = alpha * Y/K - delta
%
% Important note (timing convention):
% At i-th row, Capital is the result of investment decision at i-th row
% State variables: k(-1), A, L

var ConsumptionPerCapita            $ConsumptionPerCapita$ (long_name='consumption per capita')
    CapitalPerCapita                $CapitalPerCapita$ (long_name='capital per capita')
    OutputPerCapita                 $OutputPerCapita$ (long_name='output per capita')
    InvestmentPerCapita             $InvestmentPerCapita$ (long_name='investment per capita')
    ConsumptionPerEffectiveLabor    $ConsumptionPerEffectiveLabor$ (long_name='consumption per effective labor')
    CapitalPerEffectiveLabor        $CapitalPerEffectiveLabor$ (long_name='capital per effective labor')
    OutputPerEffectiveLabor         $OutputPerEffectiveLabor$ (long_name='output per effective labor')
    InvestmentPerEffectiveLabor     $InvestmentPerEffectiveLabor$ (long_name='investment per effective labor')
    Labor                           $Labor$ (long_name='labor/population')
    Technology                      $Technology$ (long_name='technology level')
    InterestRate                    $InterestRate$ (long_name='real interest rate')
    GrossReturn                     $GrossReturn$ (long_name='gross return on capital')
    MarginalProductCapital          $MarginalProductCapital$ (long_name='marginal product of capital')
    WagePerEffectiveLabor           $WagePerEffectiveLabor$ (long_name='wage per effective labor unit')
    Consumption                     $Consumption$ (long_name='aggregate consumption')
    Capital                         $Capital$ (long_name='aggregate capital')
    Output                          $Output$ (long_name='aggregate output');

parameters alpha beta delta sigma n g
           k_tilde_ss c_tilde_ss y_tilde_ss i_tilde_ss r_ss
           start_capital start_labor start_technology;

% Parameter defaults
@#if !defined(alpha)
  @#define alpha = 0.33
@#endif

@#if !defined(beta)
  @#define beta = 0.96
@#endif

@#if !defined(delta)
  @#define delta = 0.1
@#endif

@#if !defined(sigma)
  @#define sigma = 1.0
@#endif

@#if !defined(n)
  @#define n = 0.01
@#endif

@#if !defined(g)
  @#define g = 0.02
@#endif

@#if !defined(start_capital)
  @#define start_capital = 1.0
@#endif

@#if !defined(start_labor)
  @#define start_labor = 1.0
@#endif

@#if !defined(start_technology)
  @#define start_technology = 1.0
@#endif

alpha = @{alpha};
beta = @{beta};
delta = @{delta};
sigma = @{sigma};
n = @{n};
g = @{g};
start_capital = @{start_capital};
start_labor = @{start_labor};
start_technology = @{start_technology};

r_ss = (1 + g)^sigma / beta - 1;                           % Steady state interest rate
k_tilde_ss = (alpha / (r_ss + delta))^(1/(1 - alpha));     % Capital per effective labor
y_tilde_ss = k_tilde_ss^alpha;                             % Output per effective labor
i_tilde_ss = (delta + n + g + n*g) * k_tilde_ss;           % Investment per effective labor
c_tilde_ss = y_tilde_ss - i_tilde_ss;                      % Consumption per effective labor

model;

[name='Population/Labor dynamics']
Labor = (1 + n) * Labor(-1);

[name='Technological progress']
Technology = (1 + g) * Technology(-1);

[name='Production function (per effective labor)']
OutputPerEffectiveLabor = CapitalPerEffectiveLabor(-1)^alpha;

[name='Resource constraint (per effective labor)']
OutputPerEffectiveLabor = ConsumptionPerEffectiveLabor + InvestmentPerEffectiveLabor;

[name='Capital accumulation (per effective labor)']
CapitalPerEffectiveLabor = ((1 - delta) * CapitalPerEffectiveLabor(-1) + InvestmentPerEffectiveLabor) / ((1 + g) * (1 + n));

[name='Marginal product of capital']
MarginalProductCapital = alpha * CapitalPerEffectiveLabor(-1)^(alpha - 1);

[name='Interest rate (net return)']
InterestRate = MarginalProductCapital - delta;

[name='Gross return on capital']
GrossReturn = 1 + InterestRate;

[name='Wage per effective labor unit']
WagePerEffectiveLabor = (1 - alpha) * CapitalPerEffectiveLabor(-1)^alpha;

[name='Euler equation CRRA (per effective labor)']
ConsumptionPerEffectiveLabor^(-sigma) = beta * ConsumptionPerEffectiveLabor(+1)^(-sigma) * (1 + g)^(-sigma) * GrossReturn(+1);

[name='Per capita from per effective labor']
ConsumptionPerCapita = ConsumptionPerEffectiveLabor * Technology;
CapitalPerCapita = CapitalPerEffectiveLabor * Technology;
OutputPerCapita = OutputPerEffectiveLabor * Technology;
InvestmentPerCapita = InvestmentPerEffectiveLabor * Technology;

[name='Aggregate variables']
Consumption = ConsumptionPerCapita * Labor;
Capital = CapitalPerCapita * Labor;
Output = OutputPerCapita * Labor;

end;

initval;
    Labor = start_labor;
    Technology = start_technology;
    CapitalPerEffectiveLabor = start_capital;
    OutputPerEffectiveLabor = start_capital^alpha;
    MarginalProductCapital = alpha * start_capital^(alpha - 1);
    InterestRate = MarginalProductCapital - delta;
    GrossReturn = 1 + InterestRate;
    WagePerEffectiveLabor = (1 - alpha) * start_capital^alpha;
    InvestmentPerEffectiveLabor = (delta + n + g + n*g) * start_capital;
    ConsumptionPerEffectiveLabor = OutputPerEffectiveLabor - InvestmentPerEffectiveLabor;
    ConsumptionPerCapita = ConsumptionPerEffectiveLabor * Technology;
    CapitalPerCapita = CapitalPerEffectiveLabor * Technology;
    OutputPerCapita = OutputPerEffectiveLabor * Technology;
    InvestmentPerCapita = InvestmentPerEffectiveLabor * Technology;
    Consumption = ConsumptionPerCapita * Labor;
    Capital = CapitalPerCapita * Labor;
    Output = OutputPerCapita * Labor;
end;

endval;
  Labor = start_labor * (1 + n)^(@{periods});
  Technology = start_technology * (1 + g)^(@{periods});
  CapitalPerEffectiveLabor = k_tilde_ss;
  OutputPerEffectiveLabor = y_tilde_ss;
  ConsumptionPerEffectiveLabor = c_tilde_ss;
  InvestmentPerEffectiveLabor = i_tilde_ss;
  MarginalProductCapital = alpha * k_tilde_ss^(alpha - 1);
  InterestRate = r_ss;
  GrossReturn = 1 + r_ss;
  WagePerEffectiveLabor = (1 - alpha) * k_tilde_ss^alpha;
  ConsumptionPerCapita = c_tilde_ss * Technology;
  CapitalPerCapita = k_tilde_ss * Technology;
  OutputPerCapita = y_tilde_ss * Technology;
  InvestmentPerCapita = i_tilde_ss * Technology;
  Consumption = ConsumptionPerCapita * Labor;
  Capital = CapitalPerCapita * Labor;
  Output = OutputPerCapita * Labor;
end;

perfect_foresight_setup(periods=@{periods});
perfect_foresight_solver;

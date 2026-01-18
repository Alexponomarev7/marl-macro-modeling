% Ramsey Model with CRRA Utility and Population Growth
% 
% Timing convention:
%   - CapitalPerCapita_t is end-of-period capital per capita
%   - Production at t uses CapitalPerCapita_{t-1}
%   - State: CapitalPerCapita(-1), Labor, OutputPerCapita
%
% Equations:
%   1. Labor dynamics:       Labor_t = (1+n) * Labor_{t-1}
%   2. Production:           OutputPerCapita_t = CapitalPerCapita_{t-1}^alpha
%   3. Resource constraint:  OutputPerCapita_t = ConsumptionPerCapita_t + InvestmentPerCapita_t
%   4. Capital accumulation: CapitalPerCapita_t = (1-delta) * CapitalPerCapita_{t-1} / (1+n) + InvestmentPerCapita_t
%   5. Euler (CRRA):         c_t^(-sigma) = beta * c_{t+1}^(-sigma) * (alpha * k_t^{alpha-1} + (1-delta) / (1+n))
%
% CRRA utility: U(c) = c ^ (1 - sigma) / (1 - sigma)
% Marginal utility: U'(c) = c ^ (-sigma)

var ConsumptionPerCapita    $ConsumptionPerCapita$ (long_name='consumption per capita')
    CapitalPerCapita        $CapitalPerCapita$ (long_name='capital per capita')
    OutputPerCapita         $OutputPerCapita$ (long_name='output per capita')
    InvestmentPerCapita     $InvestmentPerCapita$ (long_name='investment per capita')
    Labor                   $Labor$ (long_name='labor/population')
    Consumption             $Consumption$ (long_name='aggregate consumption')
    Capital                 $Capital$ (long_name='aggregate capital')
    Output                  $Output$ (long_name='aggregate output');

parameters alpha beta delta sigma n 
           k_ss c_ss y_ss i_ss 
           start_capital start_labor;

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

@#if !defined(start_capital)
  @#define start_capital = 1
@#endif

@#if !defined(start_labor)
  @#define start_labor = 1.0
@#endif

alpha = @{alpha};
beta = @{beta};
delta = @{delta};
sigma = @{sigma};
n = @{n};
start_capital = @{start_capital};
start_labor = @{start_labor};

% Steady state calculations
k_ss = ((1/beta - (1 - delta) / (1 + n)) / alpha)^(1/(alpha - 1));
y_ss = k_ss^alpha;
i_ss = k_ss * (delta + n) / (1 + n);
c_ss = y_ss - i_ss;

model;

[name='Population/Labor dynamics']
Labor = (1 + n) * Labor(-1);

[name='Production function (per capita)']
OutputPerCapita = CapitalPerCapita(-1)^alpha;

[name='Resource constraint (per capita)']
OutputPerCapita = ConsumptionPerCapita + InvestmentPerCapita;

[name='Capital accumulation (per capita)']
CapitalPerCapita = (1 - delta) * CapitalPerCapita(-1) / (1 + n) + InvestmentPerCapita;

[name='Euler equation CRRA']
// MU(c_t) = beta * MU(c_{t+1}) * (MPK_{t+1} + 1 - delta)
// c_t^(-sigma) = beta * c_{t+1}^(-sigma) * (alpha * k_t^(alpha-1) + (1-delta) / (1+n))
ConsumptionPerCapita^(-sigma) = beta * ConsumptionPerCapita(+1)^(-sigma) * (alpha * CapitalPerCapita^(alpha-1) + (1-delta)/(1+n));

[name='Aggregate consumption']
Consumption = ConsumptionPerCapita * Labor;

[name='Aggregate capital']
Capital = CapitalPerCapita * Labor;

[name='Aggregate output']
Output = OutputPerCapita * Labor;

end;

initval;
  Labor = start_labor;
  CapitalPerCapita = start_capital;
  OutputPerCapita = start_capital^alpha;
  InvestmentPerCapita = (delta + n) / (1 + n) * start_capital;
  ConsumptionPerCapita = OutputPerCapita - InvestmentPerCapita;

  Consumption = ConsumptionPerCapita * Labor;
  Capital = CapitalPerCapita * Labor;
  Output = OutputPerCapita * Labor;
end;

endval;
  Labor = start_labor * (1 + n)^(@{periods});
  CapitalPerCapita = k_ss;
  OutputPerCapita = y_ss;
  ConsumptionPerCapita = c_ss;
  InvestmentPerCapita = i_ss;

  Consumption = ConsumptionPerCapita * Labor;
  Capital = CapitalPerCapita * Labor;
  Output = OutputPerCapita * Labor;
end;

perfect_foresight_setup(periods=@{periods});
perfect_foresight_solver;

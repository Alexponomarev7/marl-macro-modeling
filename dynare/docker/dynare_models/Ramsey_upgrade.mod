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

% Only the stationary per-effective-labor system is solved; levels are x_t = x_tilde_t * (1+g)^t * (1+n)^t.
var ConsumptionPerEffectiveLabor    $ConsumptionPerEffectiveLabor$ (long_name='consumption per effective labor')
    CapitalPerEffectiveLabor        $CapitalPerEffectiveLabor$ (long_name='capital per effective labor')
    OutputPerEffectiveLabor         $OutputPerEffectiveLabor$ (long_name='output per effective labor')
    InvestmentPerEffectiveLabor     $InvestmentPerEffectiveLabor$ (long_name='investment per effective labor')
    InterestRate                    $InterestRate$ (long_name='real interest rate')
    MarginalProductCapital          $MarginalProductCapital$ (long_name='marginal product of capital')
    WagePerEffectiveLabor           $WagePerEffectiveLabor$ (long_name='wage per effective labor unit');

parameters alpha beta delta sigma n g
           k_tilde_ss c_tilde_ss y_tilde_ss i_tilde_ss r_ss
           start_capital;

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

alpha = @{alpha};
beta = @{beta};
delta = @{delta};
sigma = @{sigma};
n = @{n};
g = @{g};
start_capital = @{start_capital};

r_ss = (1 + g)^sigma / beta - 1;                           % Steady state interest rate
k_tilde_ss = (alpha / (r_ss + delta))^(1/(1 - alpha));     % Capital per effective labor

% initial capital = start_capital_ratio * steady-state capital
@#if defined(start_capital_ratio)
start_capital = @{start_capital_ratio} * k_tilde_ss;
@#endif
y_tilde_ss = k_tilde_ss^alpha;                             % Output per effective labor
i_tilde_ss = (delta + n + g + n*g) * k_tilde_ss;           % Investment per effective labor
c_tilde_ss = y_tilde_ss - i_tilde_ss;                      % Consumption per effective labor

model;

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

[name='Wage per effective labor unit']
WagePerEffectiveLabor = (1 - alpha) * CapitalPerEffectiveLabor(-1)^alpha;

[name='Euler equation CRRA (per effective labor)']
ConsumptionPerEffectiveLabor^(-sigma) = beta * ConsumptionPerEffectiveLabor(+1)^(-sigma) * (1 + g)^(-sigma) * (1 + InterestRate(+1));

end;

initval;
    CapitalPerEffectiveLabor = start_capital;
    OutputPerEffectiveLabor = start_capital^alpha;
    MarginalProductCapital = alpha * start_capital^(alpha - 1);
    InterestRate = MarginalProductCapital - delta;
    WagePerEffectiveLabor = (1 - alpha) * start_capital^alpha;
    InvestmentPerEffectiveLabor = (delta + n + g + n*g) * start_capital;
    ConsumptionPerEffectiveLabor = OutputPerEffectiveLabor - InvestmentPerEffectiveLabor;
end;

endval;
  CapitalPerEffectiveLabor = k_tilde_ss;
  OutputPerEffectiveLabor = y_tilde_ss;
  ConsumptionPerEffectiveLabor = c_tilde_ss;
  InvestmentPerEffectiveLabor = i_tilde_ss;
  MarginalProductCapital = alpha * k_tilde_ss^(alpha - 1);
  InterestRate = r_ss;
  WagePerEffectiveLabor = (1 - alpha) * k_tilde_ss^alpha;
end;

perfect_foresight_setup(periods=@{periods});
perfect_foresight_solver;

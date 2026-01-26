% RBC Model with Labor and Capital Stock Shock
% Extended RBC with endogenous labor supply and capital destruction shock
%
% Timing convention:
%   - Capital_t is end-of-period t capital (chosen at t, used for production at t+1)
%   - Production at t: Y_t = A_t * K_{t-1}^alpha * L_t^{1-alpha}
%   - Capital shock destroys fraction of capital during accumulation
%
% Utility: U = C^{1-sigma} / (1-sigma) + psi * log(1-L)  [separable, log leisure]

var Consumption              $Consumption$ (long_name='consumption')
    Capital                  $Capital$ (long_name='capital stock (end of period)')
    Output                   $Output$ (long_name='output')
    Investment               $Investment$ (long_name='investment')
    Labor                    $Labor$ (long_name='hours worked')
    MarginalProductCapital   $MarginalProductCapital$ (long_name='marginal product of capital')
    InterestRate             $InterestRate$ (long_name='real interest rate')
    Wage                     $Wage$ (long_name='real wage')
    LoggedProductivity       $LoggedProductivity$ (long_name='logged TFP')
    Productivity             $Productivity$ (long_name='TFP level');

varexo LoggedProductivityInnovation   $LoggedProductivityInnovation$ (long_name='TFP shock')
       CapitalStockInnovation         $CapitalStockInnovation$ (long_name='capital destruction shock');

parameters alpha beta delta rho sigma psi
           start_capital
           k_ss y_ss c_ss i_ss l_ss r_ss w_ss;

@#if !defined(alpha)
  @#define alpha = 0.33
@#endif

@#if !defined(beta)
  @#define beta = 0.985
@#endif

@#if !defined(delta)
  @#define delta = 0.025
@#endif

@#if !defined(rho)
  @#define rho = 0.9
@#endif

@#if !defined(sigma)
  @#define sigma = 1.0
@#endif

@#if !defined(l_ss_target)
  @#define l_ss_target = 0.33
@#endif

@#if !defined(start_capital)
  @#define start_capital = 1.0
@#endif

@#if !defined(num_productivity_shocks)
  @#define num_productivity_shocks = 0
@#endif

@#if !defined(productivity_shock_period_1)
  @#define productivity_shock_period_1 = 1
@#endif
@#if !defined(productivity_shock_value_1)
  @#define productivity_shock_value_1 = 0.0
@#endif

@#if !defined(productivity_shock_period_2)
  @#define productivity_shock_period_2 = 1
@#endif
@#if !defined(productivity_shock_value_2)
  @#define productivity_shock_value_2 = 0.0
@#endif

@#if !defined(productivity_shock_period_3)
  @#define productivity_shock_period_3 = 1
@#endif
@#if !defined(productivity_shock_value_3)
  @#define productivity_shock_value_3 = 0.0
@#endif

@#if !defined(productivity_shock_period_4)
  @#define productivity_shock_period_4 = 1
@#endif
@#if !defined(productivity_shock_value_4)
  @#define productivity_shock_value_4 = 0.0
@#endif

@#if !defined(productivity_shock_period_5)
  @#define productivity_shock_period_5 = 1
@#endif
@#if !defined(productivity_shock_value_5)
  @#define productivity_shock_value_5 = 0.0
@#endif

@#if !defined(num_capital_shocks)
  @#define num_capital_shocks = 0
@#endif

@#if !defined(capital_shock_period_1)
  @#define capital_shock_period_1 = 1
@#endif
@#if !defined(capital_shock_value_1)
  @#define capital_shock_value_1 = 0.0
@#endif

@#if !defined(capital_shock_period_2)
  @#define capital_shock_period_2 = 1
@#endif
@#if !defined(capital_shock_value_2)
  @#define capital_shock_value_2 = 0.0
@#endif

@#if !defined(capital_shock_period_3)
  @#define capital_shock_period_3 = 1
@#endif
@#if !defined(capital_shock_value_3)
  @#define capital_shock_value_3 = 0.0
@#endif

@#if !defined(capital_shock_period_4)
  @#define capital_shock_period_4 = 1
@#endif
@#if !defined(capital_shock_value_4)
  @#define capital_shock_value_4 = 0.0
@#endif

@#if !defined(capital_shock_period_5)
  @#define capital_shock_period_5 = 1
@#endif
@#if !defined(capital_shock_value_5)
  @#define capital_shock_value_5 = 0.0
@#endif

alpha = @{alpha};
beta = @{beta};
delta = @{delta};
rho = @{rho};
sigma = @{sigma};
l_ss = @{l_ss_target};
start_capital = @{start_capital};

% From Euler: r_ss = 1/beta - 1
% From MPK: r_ss = alpha * (k/l)^(alpha-1) - delta

r_ss = 1 / beta - 1;
k_ss = ((r_ss + delta) / alpha) ^ (1 / (alpha - 1)) * l_ss;
y_ss = k_ss^alpha * l_ss^(1 - alpha);
i_ss = delta * k_ss;
c_ss = y_ss - i_ss;
w_ss = (1 - alpha) * y_ss / l_ss;
psi = w_ss * (1 - l_ss) / c_ss^sigma;

model;

[name='Logged productivity law of motion']
LoggedProductivity = rho * LoggedProductivity(-1) + LoggedProductivityInnovation;

[name='Productivity level']
Productivity = exp(LoggedProductivity);

[name='Production function (Cobb-Douglas with labor)']
Output = Productivity * Capital(-1)^alpha * Labor^(1 - alpha);

[name='Resource constraint']
Output = Consumption + Investment;

[name='Capital accumulation with destruction shock']
% CapitalStockInnovation more 0 means capital destruction
Capital = exp(-CapitalStockInnovation) * (1 - delta) * Capital(-1) + Investment;

[name='Real wage (MPL)']
Wage = (1 - alpha) * Output / Labor;

[name='Marginal product of capital (current)']
MarginalProductCapital = alpha * Productivity * Capital(-1)^(alpha - 1) * Labor^(1 - alpha);

[name='Interest rate (current MPK - delta)']
InterestRate = MarginalProductCapital - delta;

[name='Euler equation (CRRA)']
Consumption^(-sigma) = beta * Consumption(+1)^(-sigma) * (1 + alpha * Productivity(+1) * Capital^(alpha-1) * Labor(+1)^(1-alpha) - delta);

[name='Labor supply (intratemporal FOC)']
psi * Consumption^sigma / (1 - Labor) = Wage;

end;

initval;
  LoggedProductivityInnovation = 0;
  CapitalStockInnovation = 0;
  LoggedProductivity = 0;
  Productivity = 1;
  Capital = start_capital;
  Labor = l_ss;
  Output = start_capital^alpha * l_ss^(1 - alpha);
  Investment = delta * start_capital;
  Consumption = Output - Investment;
  Wage = (1 - alpha) * Output / l_ss;
  MarginalProductCapital = alpha * start_capital^(alpha-1) * l_ss^(1-alpha);
  InterestRate = alpha * start_capital^(alpha - 1) * l_ss^(1 - alpha) - delta;
end;

endval;
  LoggedProductivityInnovation = 0;
  CapitalStockInnovation = 0;
  LoggedProductivity = 0;
  Productivity = 1;
  Capital = k_ss;
  Output = y_ss;
  Consumption = c_ss;
  Investment = i_ss;
  Labor = l_ss;
  Wage = w_ss;
  MarginalProductCapital = alpha * k_ss^(alpha-1) * l_ss^(1-alpha);
  InterestRate = r_ss;
end;

shocks;
  var LoggedProductivityInnovation;
  periods @{productivity_shock_period_1} @{productivity_shock_period_2} @{productivity_shock_period_3} @{productivity_shock_period_4} @{productivity_shock_period_5};
  values @{productivity_shock_value_1} @{productivity_shock_value_2} @{productivity_shock_value_3} @{productivity_shock_value_4} @{productivity_shock_value_5};

  var CapitalStockInnovation;
  periods @{capital_shock_period_1} @{capital_shock_period_2} @{capital_shock_period_3} @{capital_shock_period_4} @{capital_shock_period_5};
  values @{capital_shock_value_1} @{capital_shock_value_2} @{capital_shock_value_3} @{capital_shock_value_4} @{capital_shock_value_5};
end;

perfect_foresight_setup(periods=@{periods});
perfect_foresight_solver;

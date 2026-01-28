% RBC Model with Productivity Shocks
% Based on standard Real Business Cycle framework
%
% Timing convention:
%   - Capital_t is end-of-period t capital (chosen at t, used for production at t+1)
%   - Production at t: Y_t = A_t * K_{t-1}^alpha
%   - Euler: u'(C_t) = beta * u'(C_{t+1}) * [alpha * A_{t+1} * K_t^{alpha-1} + 1 - delta]

var Consumption         $Consumption$ (long_name='consumption')
    Capital             $Capital$ (long_name='capital')
    Output              $Output$ (long_name='output')
    Investment          $Investment$ (long_name='investment')
    InterestRate        $InterestRate$ (long_name='real interest rate')
    LoggedProductivity  $LoggedProductivity$ (long_name='logged TFP')
    Productivity        $Productivity$ (long_name='TFP level');

varexo LoggedProductivityInnovation $LoggedProductivityInnovation$ (long_name='TFP shock');

parameters alpha beta delta rho sigma
           start_capital
           k_ss c_ss y_ss r_ss;

@#if !defined(alpha)
  @#define alpha = 0.33
@#endif

@#if !defined(beta)
  @#define beta = 0.985
@#endif

@#if !defined(delta)
  @#define delta = 0.033
@#endif

@#if !defined(rho)
  @#define rho = 0.9
@#endif

@#if !defined(sigma)
  @#define sigma = 1.0
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

@#ifndef periods
    @#define periods = 100
@#endif

alpha = @{alpha};
beta = @{beta};
delta = @{delta};
rho = @{rho};
sigma = @{sigma};
start_capital = @{start_capital};

k_ss = ((1 / beta - (1 - delta)) / alpha) ^ (1 / (alpha - 1));
y_ss = k_ss^alpha;  % A = 1 in steady state
i_ss = delta * k_ss;
c_ss = y_ss - i_ss;
r_ss = 1 / beta - 1;  % Или эквивалентно: alpha * k_ss^(alpha-1) - delta

model;

[name='Logged productivity law of motion']
LoggedProductivity = rho * LoggedProductivity(-1) + LoggedProductivityInnovation;

[name='Productivity level']
Productivity = exp(LoggedProductivity);

[name='Production function']
Output = Productivity * Capital(-1)^alpha;

[name='Resource constraint']
Output = Consumption + Investment;

[name='Capital accumulation']
Capital = (1 - delta) * Capital(-1) + Investment;

[name='Interest rate (MPK - delta)']
InterestRate = alpha * Productivity * Capital(-1)^(alpha - 1) - delta;

[name='Euler equation (CRRA)']
Consumption^(-sigma) = beta * Consumption(+1)^(-sigma) * (1 + InterestRate);

end;

initval;
  LoggedProductivityInnovation = 0;
  LoggedProductivity = 0;
  Productivity = 1;
  Capital = start_capital;
  Output = Capital^alpha;
  Investment = delta * Capital;
  Consumption = Output - Investment;
  InterestRate = alpha * Capital ^ (alpha - 1) - delta;
end;

endval;
  LoggedProductivityInnovation = 0;
  LoggedProductivity = 0;
  Productivity = 1;
  Capital = k_ss;
  Output = y_ss;
  Consumption = c_ss;
  Investment = i_ss;
  InterestRate = r_ss;
end;

shocks;
  var LoggedProductivityInnovation;
  periods @{productivity_shock_period_1} @{productivity_shock_period_2} @{productivity_shock_period_3} @{productivity_shock_period_4} @{productivity_shock_period_5};
  values @{productivity_shock_value_1} @{productivity_shock_value_2} @{productivity_shock_value_3} @{productivity_shock_value_4} @{productivity_shock_value_5};
end;

perfect_foresight_setup(periods=@{periods});
perfect_foresight_solver;

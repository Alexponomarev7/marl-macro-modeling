% RBC Model with Productivity Shocks
% Based on standard Real Business Cycle framework
%
% Timing convention:
%   - Capital_t is end-of-period t capital (chosen at t, used for production at t+1)
%   - Production at t: Y_t = A_t * K_{t-1}^alpha
%   - Euler: u'(C_t) = β * u'(C_{t+1}) * [α * A_{t+1} * K_t^{α-1} + 1 - δ]

var Consumption         $Consumption$ (long_name='consumption')
    Capital             $Capital$ (long_name='capital')
    Output              $Output$ (long_name='output')
    Investment          $Investment$ (long_name='investment')
    LoggedProductivity  $LoggedProductivity$ (long_name='logged TFP')
    Productivity        $Productivity$ (long_name='TFP level')
    InterestRate        $InterestRate$ (long_name='real interest rate');

varexo LoggedProductivityInnovation $LoggedProductivityInnovation$ (long_name='TFP shock');

parameters alpha beta delta rho sigma k_ss c_ss y_ss r_ss;

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

@#if !defined(productivity_shock_period_6)
  @#define productivity_shock_period_6 = 1
@#endif
@#if !defined(productivity_shock_value_6)
  @#define productivity_shock_value_6 = 0.0
@#endif

alpha = @{alpha};
beta = @{beta};
delta = @{delta};
rho = @{rho};
sigma = @{sigma};

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
InterestRate = alpha * Productivity(+1) * Capital^(alpha - 1) - delta;

[name='Euler equation (CRRA)']
Consumption^(-sigma) = beta * Consumption(+1)^(-sigma) * (1 + InterestRate);

end;

initval;
  LoggedProductivityInnovation = 0;
  LoggedProductivity = 0;
  Productivity = 1;
  Capital = k_ss;
  Output = y_ss;
  Consumption = c_ss;
  Investment = delta * k_ss;
  InterestRate = r_ss;
end;

endval;
  LoggedProductivityInnovation = 0;
  LoggedProductivity = 0;
  Productivity = 1;
  Capital = k_ss;
  Output = y_ss;
  Consumption = c_ss;
  Investment = delta * k_ss;
  InterestRate = r_ss;
end;

shocks;
  var LoggedProductivityInnovation;

@#if num_productivity_shocks == 1
  periods @{productivity_shock_period_1};
  values @{productivity_shock_value_1};
@#endif

@#if num_productivity_shocks == 2
  periods @{productivity_shock_period_1} @{productivity_shock_period_2};
  values @{productivity_shock_value_1} @{productivity_shock_value_2};
@#endif

@#if num_productivity_shocks == 3
  periods @{shock_period_1} @{productivity_shock_period_2} @{productivity_shock_period_3};
  values @{productivity_shock_value_1} @{productivity_shock_value_2} @{productivity_shock_value_3};
@#endif

@#if num_productivity_shocks == 4
  periods @{productivity_shock_period_1} @{productivity_shock_period_2} @{productivity_shock_period_3} @{productivity_shock_period_4};
  values @{productivity_shock_value_1} @{productivity_shock_value_2} @{productivity_shock_value_3} @{productivity_shock_value_4};
@#endif

@#if num_productivity_shocks == 5
  periods @{productivity_shock_period_1} @{productivity_shock_period_2} @{productivity_shock_period_3} @{productivity_shock_period_4} @{productivity_shock_period_5};
  values @{productivity_shock_value_1} @{productivity_shock_value_2} @{productivity_shock_value_3} @{productivity_shock_value_4} @{productivity_shock_value_5};
@#endif

@#if num_productivity_shocks == 6
  periods @{productivity_shock_period_1} @{productivity_shock_period_2} @{productivity_shock_period_3} @{productivity_shock_period_4} @{productivity_shock_period_5} @{productivity_shock_period_6};
  values @{productivity_shock_value_1} @{productivity_shock_value_2} @{productivity_shock_value_3} @{productivity_shock_value_4} @{productivity_shock_value_5} @{productivity_shock_value_6};
@#endif

end;

perfect_foresight_setup(periods=@{periods});
perfect_foresight_solver;

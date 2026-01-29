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

@#for i in 6:50
@#if !defined(productivity_shock_period_@{i})
  @#define productivity_shock_period_@{i} = 1
@#endif
@#if !defined(productivity_shock_value_@{i})
  @#define productivity_shock_value_@{i} = 0.0
@#endif
@#endfor

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
  periods @{productivity_shock_period_1} @{productivity_shock_period_2} @{productivity_shock_period_3} @{productivity_shock_period_4} @{productivity_shock_period_5}
          @{productivity_shock_period_6} @{productivity_shock_period_7} @{productivity_shock_period_8} @{productivity_shock_period_9} @{productivity_shock_period_10}
          @{productivity_shock_period_11} @{productivity_shock_period_12} @{productivity_shock_period_13} @{productivity_shock_period_14} @{productivity_shock_period_15}
          @{productivity_shock_period_16} @{productivity_shock_period_17} @{productivity_shock_period_18} @{productivity_shock_period_19} @{productivity_shock_period_20}
          @{productivity_shock_period_21} @{productivity_shock_period_22} @{productivity_shock_period_23} @{productivity_shock_period_24} @{productivity_shock_period_25}
          @{productivity_shock_period_26} @{productivity_shock_period_27} @{productivity_shock_period_28} @{productivity_shock_period_29} @{productivity_shock_period_30}
          @{productivity_shock_period_31} @{productivity_shock_period_32} @{productivity_shock_period_33} @{productivity_shock_period_34} @{productivity_shock_period_35}
          @{productivity_shock_period_36} @{productivity_shock_period_37} @{productivity_shock_period_38} @{productivity_shock_period_39} @{productivity_shock_period_40}
          @{productivity_shock_period_41} @{productivity_shock_period_42} @{productivity_shock_period_43} @{productivity_shock_period_44} @{productivity_shock_period_45}
          @{productivity_shock_period_46} @{productivity_shock_period_47} @{productivity_shock_period_48} @{productivity_shock_period_49} @{productivity_shock_period_50};
  values @{productivity_shock_value_1} @{productivity_shock_value_2} @{productivity_shock_value_3} @{productivity_shock_value_4} @{productivity_shock_value_5}
         @{productivity_shock_value_6} @{productivity_shock_value_7} @{productivity_shock_value_8} @{productivity_shock_value_9} @{productivity_shock_value_10}
         @{productivity_shock_value_11} @{productivity_shock_value_12} @{productivity_shock_value_13} @{productivity_shock_value_14} @{productivity_shock_value_15}
         @{productivity_shock_value_16} @{productivity_shock_value_17} @{productivity_shock_value_18} @{productivity_shock_value_19} @{productivity_shock_value_20}
         @{productivity_shock_value_21} @{productivity_shock_value_22} @{productivity_shock_value_23} @{productivity_shock_value_24} @{productivity_shock_value_25}
         @{productivity_shock_value_26} @{productivity_shock_value_27} @{productivity_shock_value_28} @{productivity_shock_value_29} @{productivity_shock_value_30}
         @{productivity_shock_value_31} @{productivity_shock_value_32} @{productivity_shock_value_33} @{productivity_shock_value_34} @{productivity_shock_value_35}
         @{productivity_shock_value_36} @{productivity_shock_value_37} @{productivity_shock_value_38} @{productivity_shock_value_39} @{productivity_shock_value_40}
         @{productivity_shock_value_41} @{productivity_shock_value_42} @{productivity_shock_value_43} @{productivity_shock_value_44} @{productivity_shock_value_45}
         @{productivity_shock_value_46} @{productivity_shock_value_47} @{productivity_shock_value_48} @{productivity_shock_value_49} @{productivity_shock_value_50};
end;

perfect_foresight_setup(periods=@{periods});
perfect_foresight_solver;

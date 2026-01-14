% Ramsey Model - Basic Version
% 
% Timing convention:
%   - Capital_t is end-of-period capital (result of investment at t)
%   - Production at t uses Capital_{t-1}
%   - State: Capital(-1), Output
%
% Equations:
%   1. Capital accumulation: Capital_t = (1-delta) * Capital_{t-1} + Investment_t
%   2. Resource constraint:  Output_t = Consumption_t + Investment_t
%   3. Production function:  Output_t = Capital_{t-1}^alpha
%   4. Euler equation:       Consumption_{t+1} = Consumption_t * beta * (alpha*Output_{t+1}/Capital_t + 1-delta)

var Consumption         $Consumption$ (long_name='consumption')
    Capital             $Capital$ (long_name='capital')
    Output              $Output$ (long_name='output')
    Investment          $Investment$ (long_name='investment');

parameters alpha beta delta k_ss c_ss y_ss i_ss start_capital;

@#if !defined(alpha)
  @#define alpha = 0.33
@#endif

@#if !defined(beta)
  @#define beta = 0.96
@#endif

@#if !defined(delta)
  @#define delta = 0.1
@#endif

@#if !defined(start_capital)
  @#define start_capital = 1.0
@#endif

alpha = @{alpha};
beta = @{beta};
delta = @{delta};
start_capital = @{start_capital};

k_ss = ((1 / beta - (1 - delta)) / alpha) ^ (1 / (alpha - 1));
y_ss = k_ss^alpha;
i_ss = delta * k_ss;
c_ss = y_ss - i_ss;

model;

[name='Capital accumulation']
Capital = (1 - delta) * Capital(-1) + Investment;

[name='Resource constraint']
Output = Consumption + Investment;

[name='Production function']
Output = Capital(-1)^alpha;

[name='Euler equation']
1/Consumption = beta * 1/Consumption(+1) * (alpha * Capital ^ (alpha - 1) + 1 - delta);

end;

% Initial conditions
initval;
  Capital = start_capital;
  Output = start_capital^alpha;
  Investment = delta * start_capital;
  Consumption = Output - Investment;
end;

% Terminal conditions (steady state)
endval;
  Capital = k_ss;
  Output = y_ss;
  Consumption = c_ss;
  Investment = i_ss;
end;

perfect_foresight_setup(periods=@{periods});
perfect_foresight_solver;

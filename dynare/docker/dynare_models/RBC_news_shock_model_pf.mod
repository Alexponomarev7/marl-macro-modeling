% RBC Model with News Shocks
% Extended RBC with anticipated and unanticipated TFP shocks
%
% Key feature: News shock becomes effective 8 periods after announcement
%   z_t = rho_z * z_{t-1} + eps_surprise_t + eps_news_{t-8}
%
% Model features:
%   - Cobb-Douglas production: Y = exp(z) * K^alpha * L^(1-alpha)
%   - CRRA utility with log leisure: U = C^(1-sigma)/(1-sigma) + psi*log(1-L)
%   - Balanced growth path with population and technology growth
%   - Variables are in LOG levels (Output = log(Y), etc.)
%
% Timing convention:
%   - Capital_t is end-of-period t capital
%   - Production at t uses Capital_{t-1}
%
% Growth:
%   - Population grows at rate n (quarterly)
%   - Technology grows at rate x (quarterly)
%   - Composite growth: gamma_x = (1+n)*(1+x)

var Output              $Output$ (long_name='output (log)')
    Consumption         $Consumption$ (long_name='consumption (log)')
    Capital             $Capital$ (long_name='capital (log)')
    Labor               $Labor$ (long_name='hours worked (log)')
    LoggedProductivity  $LoggedProductivity$ (long_name='TFP (log)')
    AnnualInterestRate  $AnnualInterestRate$ (long_name='annualized interest rate')
    Wage                $Wage$ (long_name='real wage (log)')
    Investment          $Investment$ (long_name='investment (log)');

varexo NewsShock        $NewsShock$ (long_name='anticipated TFP shock')
       SurpriseShock    $SurpriseShock$ (long_name='unanticipated TFP shock');

parameters alpha        $alpha$ (long_name='capital share in production')
           beta         $beta$ (long_name='discount factor')
           delta        $delta$ (long_name='depreciation rate')
           sigma        $sigma$ (long_name='risk aversion (CRRA parameter)')
           rhoz         $rho_z$ (long_name='TFP shock persistence')
           psi          $psi$ (long_name='labor disutility weight')
           gammax       $gamma_x$ (long_name='composite growth rate (1+n)(1+x)')
           n            $n$ (long_name='population growth rate (quarterly)')
           x            $x$ (long_name='technology growth rate (quarterly)')
           i_y          $i_y$ (long_name='investment-output ratio')
           k_y          $k_y$ (long_name='capital-output ratio')
           l_ss         $l_ss$ (long_name='steady state labor')
           k_ss         $k_ss$ (long_name='steady state capital')
           y_ss         $y_ss$ (long_name='steady state output')
           c_ss         $c_ss$ (long_name='steady state consumption')
           i_ss         $i_ss$ (long_name='steady state investment')
           w_ss         $w_ss$ (long_name='steady state wage')
           r_ss         $r_ss$ (long_name='steady state interest rate')
           start_capital_ratio;

@#if !defined(alpha)
  @#define alpha = 0.33
@#endif

@#if !defined(sigma)
  @#define sigma = 1.0
@#endif

@#if !defined(i_y)
  @#define i_y = 0.25
@#endif

@#if !defined(k_y)
  @#define k_y = 10.4
@#endif

@#if !defined(x)
  @#define x = 0.0055
@#endif

@#if !defined(n)
  @#define n = 0.0027
@#endif

@#if !defined(rhoz)
  @#define rhoz = 0.97
@#endif

@#if !defined(l_ss_target)
  @#define l_ss_target = 0.33
@#endif

@#if !defined(start_capital_ratio)
  @#define start_capital_ratio = 1.0
@#endif

@#if !defined(num_news_shocks)
  @#define num_news_shocks = 0
@#endif

@#if !defined(news_shock_period_1)
  @#define news_shock_period_1 = 1
@#endif
@#if !defined(news_shock_value_1)
  @#define news_shock_value_1 = 0.0
@#endif

@#if !defined(news_shock_period_2)
  @#define news_shock_period_2 = 1
@#endif
@#if !defined(news_shock_value_2)
  @#define news_shock_value_2 = 0.0
@#endif

@#if !defined(news_shock_period_3)
  @#define news_shock_period_3 = 1
@#endif
@#if !defined(news_shock_value_3)
  @#define news_shock_value_3 = 0.0
@#endif

@#if !defined(news_shock_period_4)
  @#define news_shock_period_4 = 1
@#endif
@#if !defined(news_shock_value_4)
  @#define news_shock_value_4 = 0.0
@#endif

@#if !defined(news_shock_period_5)
  @#define news_shock_period_5 = 1
@#endif
@#if !defined(news_shock_value_5)
  @#define news_shock_value_5 = 0.0
@#endif

// Parameters for news shocks 6-50 must be passed via -D flags from command line
// Python code always provides all 50 shock parameters

@#if !defined(num_surprise_shocks)
  @#define num_surprise_shocks = 0
@#endif

@#if !defined(surprise_shock_period_1)
  @#define surprise_shock_period_1 = 1
@#endif
@#if !defined(surprise_shock_value_1)
  @#define surprise_shock_value_1 = 0.0
@#endif

@#if !defined(surprise_shock_period_2)
  @#define surprise_shock_period_2 = 1
@#endif
@#if !defined(surprise_shock_value_2)
  @#define surprise_shock_value_2 = 0.0
@#endif

@#if !defined(surprise_shock_period_3)
  @#define surprise_shock_period_3 = 1
@#endif
@#if !defined(surprise_shock_value_3)
  @#define surprise_shock_value_3 = 0.0
@#endif

@#if !defined(surprise_shock_period_4)
  @#define surprise_shock_period_4 = 1
@#endif
@#if !defined(surprise_shock_value_4)
  @#define surprise_shock_value_4 = 0.0
@#endif

@#if !defined(surprise_shock_period_5)
  @#define surprise_shock_period_5 = 1
@#endif
@#if !defined(surprise_shock_value_5)
  @#define surprise_shock_value_5 = 0.0
@#endif

// Parameters for surprise shocks 6-50 must be passed via -D flags from command line
// Python code always provides all 50 shock parameters

% Assign parameters
alpha = @{alpha};
sigma = @{sigma};
i_y = @{i_y};
k_y = @{k_y};
x = @{x};
n = @{n};
rhoz = @{rhoz};
l_ss = @{l_ss_target};
start_capital_ratio = @{start_capital_ratio};

% Derived parameters
gammax = (1 + n) * (1 + x);
delta = i_y / k_y - x - n - n * x;
beta = gammax / (alpha / k_y + (1 - delta));

% Steady state calculations
k_ss = ((1 / beta * gammax - (1 - delta)) / alpha)^(1 / (alpha - 1)) * l_ss;
y_ss = k_ss^alpha * l_ss^(1 - alpha);
i_ss = (x + n + delta + n * x) * k_ss;
c_ss = y_ss - i_ss;
w_ss = (1 - alpha) * y_ss / l_ss;
r_ss = 4 * alpha * y_ss / k_ss;
psi = (1 - alpha) * (k_ss / l_ss)^alpha * (1 - l_ss) / c_ss^sigma;

model;

[name='Euler equation (with growth)']
exp(Consumption)^(-sigma) = beta / gammax * exp(Consumption(+1))^(-sigma) *
    (alpha * exp(LoggedProductivity(+1)) * (exp(Capital) / exp(Labor(+1)))^(alpha - 1) + (1 - delta));

[name='Labor supply FOC']
psi * exp(Consumption)^sigma / (1 - exp(Labor)) = exp(Wage);

[name='Capital accumulation (with growth)']
gammax * exp(Capital) = (1 - delta) * exp(Capital(-1)) + exp(Investment);

[name='Resource constraint']
exp(Output) = exp(Investment) + exp(Consumption);

[name='Production function']
exp(Output) = exp(LoggedProductivity) * exp(Capital(-1))^alpha * exp(Labor)^(1 - alpha);

[name='Wage equation (MPL)']
exp(Wage) = (1 - alpha) * exp(Output) / exp(Labor);

[name='Annualized interest rate']
AnnualInterestRate = 4 * alpha * exp(Output) / exp(Capital(-1));

[name='TFP process with news shock']
LoggedProductivity = rhoz * LoggedProductivity(-1) + SurpriseShock + NewsShock(-8);

end;

initval;
  NewsShock = 0;
  SurpriseShock = 0;
  LoggedProductivity = 0;
  Capital = log(start_capital_ratio * k_ss);
  Labor = log(l_ss);
  Output = alpha * log(start_capital_ratio * k_ss) + (1 - alpha) * log(l_ss);
  Investment = log((gammax - 1 + delta) * start_capital_ratio * k_ss);
  Consumption = log(exp(Output) - exp(Investment));
  Wage = log((1 - alpha) * exp(Output) / exp(Labor));
  AnnualInterestRate = 4 * alpha * exp(Output) / exp(Capital);
end;

endval;
  NewsShock = 0;
  SurpriseShock = 0;
  LoggedProductivity = 0;
  Capital = log(k_ss);
  Output = log(y_ss);
  Consumption = log(c_ss);
  Investment = log(i_ss);
  Labor = log(l_ss);
  Wage = log(w_ss);
  AnnualInterestRate = r_ss;
end;

shocks;
  var NewsShock;
  periods @{news_shock_period_1} @{news_shock_period_2} @{news_shock_period_3} @{news_shock_period_4} @{news_shock_period_5}
          @{news_shock_period_6} @{news_shock_period_7} @{news_shock_period_8} @{news_shock_period_9} @{news_shock_period_10}
          @{news_shock_period_11} @{news_shock_period_12} @{news_shock_period_13} @{news_shock_period_14} @{news_shock_period_15}
          @{news_shock_period_16} @{news_shock_period_17} @{news_shock_period_18} @{news_shock_period_19} @{news_shock_period_20}
          @{news_shock_period_21} @{news_shock_period_22} @{news_shock_period_23} @{news_shock_period_24} @{news_shock_period_25}
          @{news_shock_period_26} @{news_shock_period_27} @{news_shock_period_28} @{news_shock_period_29} @{news_shock_period_30}
          @{news_shock_period_31} @{news_shock_period_32} @{news_shock_period_33} @{news_shock_period_34} @{news_shock_period_35}
          @{news_shock_period_36} @{news_shock_period_37} @{news_shock_period_38} @{news_shock_period_39} @{news_shock_period_40}
          @{news_shock_period_41} @{news_shock_period_42} @{news_shock_period_43} @{news_shock_period_44} @{news_shock_period_45}
          @{news_shock_period_46} @{news_shock_period_47} @{news_shock_period_48} @{news_shock_period_49} @{news_shock_period_50};
  values @{news_shock_value_1} @{news_shock_value_2} @{news_shock_value_3} @{news_shock_value_4} @{news_shock_value_5}
         @{news_shock_value_6} @{news_shock_value_7} @{news_shock_value_8} @{news_shock_value_9} @{news_shock_value_10}
         @{news_shock_value_11} @{news_shock_value_12} @{news_shock_value_13} @{news_shock_value_14} @{news_shock_value_15}
         @{news_shock_value_16} @{news_shock_value_17} @{news_shock_value_18} @{news_shock_value_19} @{news_shock_value_20}
         @{news_shock_value_21} @{news_shock_value_22} @{news_shock_value_23} @{news_shock_value_24} @{news_shock_value_25}
         @{news_shock_value_26} @{news_shock_value_27} @{news_shock_value_28} @{news_shock_value_29} @{news_shock_value_30}
         @{news_shock_value_31} @{news_shock_value_32} @{news_shock_value_33} @{news_shock_value_34} @{news_shock_value_35}
         @{news_shock_value_36} @{news_shock_value_37} @{news_shock_value_38} @{news_shock_value_39} @{news_shock_value_40}
         @{news_shock_value_41} @{news_shock_value_42} @{news_shock_value_43} @{news_shock_value_44} @{news_shock_value_45}
         @{news_shock_value_46} @{news_shock_value_47} @{news_shock_value_48} @{news_shock_value_49} @{news_shock_value_50};

  var SurpriseShock;
  periods @{surprise_shock_period_1} @{surprise_shock_period_2} @{surprise_shock_period_3} @{surprise_shock_period_4} @{surprise_shock_period_5}
          @{surprise_shock_period_6} @{surprise_shock_period_7} @{surprise_shock_period_8} @{surprise_shock_period_9} @{surprise_shock_period_10}
          @{surprise_shock_period_11} @{surprise_shock_period_12} @{surprise_shock_period_13} @{surprise_shock_period_14} @{surprise_shock_period_15}
          @{surprise_shock_period_16} @{surprise_shock_period_17} @{surprise_shock_period_18} @{surprise_shock_period_19} @{surprise_shock_period_20}
          @{surprise_shock_period_21} @{surprise_shock_period_22} @{surprise_shock_period_23} @{surprise_shock_period_24} @{surprise_shock_period_25}
          @{surprise_shock_period_26} @{surprise_shock_period_27} @{surprise_shock_period_28} @{surprise_shock_period_29} @{surprise_shock_period_30}
          @{surprise_shock_period_31} @{surprise_shock_period_32} @{surprise_shock_period_33} @{surprise_shock_period_34} @{surprise_shock_period_35}
          @{surprise_shock_period_36} @{surprise_shock_period_37} @{surprise_shock_period_38} @{surprise_shock_period_39} @{surprise_shock_period_40}
          @{surprise_shock_period_41} @{surprise_shock_period_42} @{surprise_shock_period_43} @{surprise_shock_period_44} @{surprise_shock_period_45}
          @{surprise_shock_period_46} @{surprise_shock_period_47} @{surprise_shock_period_48} @{surprise_shock_period_49} @{surprise_shock_period_50};
  values @{surprise_shock_value_1} @{surprise_shock_value_2} @{surprise_shock_value_3} @{surprise_shock_value_4} @{surprise_shock_value_5}
         @{surprise_shock_value_6} @{surprise_shock_value_7} @{surprise_shock_value_8} @{surprise_shock_value_9} @{surprise_shock_value_10}
         @{surprise_shock_value_11} @{surprise_shock_value_12} @{surprise_shock_value_13} @{surprise_shock_value_14} @{surprise_shock_value_15}
         @{surprise_shock_value_16} @{surprise_shock_value_17} @{surprise_shock_value_18} @{surprise_shock_value_19} @{surprise_shock_value_20}
         @{surprise_shock_value_21} @{surprise_shock_value_22} @{surprise_shock_value_23} @{surprise_shock_value_24} @{surprise_shock_value_25}
         @{surprise_shock_value_26} @{surprise_shock_value_27} @{surprise_shock_value_28} @{surprise_shock_value_29} @{surprise_shock_value_30}
         @{surprise_shock_value_31} @{surprise_shock_value_32} @{surprise_shock_value_33} @{surprise_shock_value_34} @{surprise_shock_value_35}
         @{surprise_shock_value_36} @{surprise_shock_value_37} @{surprise_shock_value_38} @{surprise_shock_value_39} @{surprise_shock_value_40}
         @{surprise_shock_value_41} @{surprise_shock_value_42} @{surprise_shock_value_43} @{surprise_shock_value_44} @{surprise_shock_value_45}
         @{surprise_shock_value_46} @{surprise_shock_value_47} @{surprise_shock_value_48} @{surprise_shock_value_49} @{surprise_shock_value_50};
end;

perfect_foresight_setup(periods=@{periods});
perfect_foresight_solver;

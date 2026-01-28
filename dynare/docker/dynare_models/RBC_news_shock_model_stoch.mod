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
           r_ss         $r_ss$ (long_name='steady state interest rate');

% Parameter defaults
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

@#if !defined(news_shock_stderr)
  @#define news_shock_stderr = 0.01
@#endif

@#if !defined(surprise_shock_stderr)
  @#define surprise_shock_stderr = 0.01
@#endif

% Assign parameters
alpha = @{alpha};
sigma = @{sigma};
i_y = @{i_y};
k_y = @{k_y};
x = @{x};
n = @{n};
rhoz = @{rhoz};
l_ss = @{l_ss_target};

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
  Output = log(y_ss);
  Capital = log(k_ss);
  Consumption = log(c_ss);
  Investment = log(i_ss);
  Labor = log(l_ss);
  Wage = log(w_ss);
  AnnualInterestRate = r_ss;
end;

steady;
check;

shocks;
  var NewsShock;
  stderr @{news_shock_stderr};
  var SurpriseShock;
  stderr @{surprise_shock_stderr};
end;

stoch_simul(order=1, periods=@{periods}, drop=0, irf=0, nomoments, nofunctions, nograph, nocorr, noprint);

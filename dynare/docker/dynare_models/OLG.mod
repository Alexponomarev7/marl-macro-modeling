% OLG Model - stochastic Diamond (1965) with growth and CRRA preferences
%
% Structure:
%   - 2 generations: Young (1) and Old (2); one period ~ one generation
%   - CRRA utility: U = u(c1_t) + beta * E_t u(c2_{t+1}),  u(c) = c^(1-sigma)/(1-sigma)
%   - TFP shocks: z_t = rho_z z_{t-1} + eps_t (log TFP), shared by wages and the return to capital
%   - Variables are normalized by effective labor (Technology * Population)
%
% Timing convention (consistent with Ramsey script):
%   - Capital refers to k_t (end-of-period stock, result of savings at t)
%   - Production at t uses Capital(-1)
%
% Equations:
%   1. Euler: c1_t^(-sigma) = beta * E_t[(1 + r_{t+1}) * c2_{t+1}^(-sigma)]
%   2. Budget Young: c1_t + s_t = w_t
%   3. Budget Old:   c2_t = (1 + r_t) * s_{t-1}
%   4. Capital Market: (1+n)(1+g) * Capital_t = Savings_t
%   5. Factor prices: w_t and r_t determined by Capital(-1) and TFP

var ConsYoung          $ConsYoung$ (long_name='consumption young')
    ConsOld            $ConsOld$ (long_name='consumption old')
    Savings            $Savings$ (long_name='savings per effective worker')
    Capital            $Capital$ (long_name='capital stock (end of period)')
    Output             $Output$ (long_name='output per effective worker')
    Wage               $Wage$ (long_name='wage rate')
    InterestRate       $InterestRate$ (long_name='rental rate of capital')
    LoggedProductivity $LoggedProductivity$ (long_name='TFP (log)');

varexo ProductivityInnovation $ProductivityInnovation$ (long_name='TFP innovation');

parameters alpha beta delta n g sigma rho_z k_ss y_ss w_ss r_ss s_ss c1_ss c2_ss;

@#if !defined(alpha)
    @#define alpha = 0.33
@#endif
@#if !defined(beta)
    @#define beta = 0.5
@#endif
@#if !defined(delta)
    @#define delta = 0.85
@#endif
@#if !defined(n)
    @#define n = 0.07
@#endif
@#if !defined(g)
    @#define g = 0.25
@#endif
@#if !defined(sigma)
    @#define sigma = 2.0
@#endif
@#if !defined(rho_z)
    @#define rho_z = 0.3
@#endif
@#if !defined(productivity_shock_stderr)
    @#define productivity_shock_stderr = 0.1
@#endif

alpha = @{alpha};
beta = @{beta};
delta = @{delta};
n = @{n};
g = @{g};
sigma = @{sigma};
rho_z = @{rho_z};

% log-utility closed form: initial guess for steady
k_ss = (beta * (1 - alpha) / ((1 + beta) * (1 + n) * (1 + g))) ^ (1 / (1 - alpha));
y_ss = k_ss^alpha;
w_ss = (1 - alpha) * k_ss^alpha;
r_ss = alpha * k_ss^(alpha - 1) - delta;
s_ss = beta / (1 + beta) * w_ss;
c1_ss = w_ss - s_ss;
c2_ss = (1 + r_ss) * s_ss;

model;

[name='Production (per effective labor)']
Output = exp(LoggedProductivity) * Capital(-1)^alpha;

[name='Wage determination (MPL)']
Wage = (1 - alpha) * exp(LoggedProductivity) * Capital(-1)^alpha;

[name='Interest Rate determination (MPK)']
InterestRate = alpha * exp(LoggedProductivity) * Capital(-1)^(alpha - 1) - delta;

[name='Euler Equation (CRRA)']
ConsYoung^(-sigma) = beta * (1 + InterestRate(+1)) * ConsOld(+1)^(-sigma);

[name='Budget Constraint - Young']
ConsYoung + Savings = Wage;

[name='Budget Constraint - Old']
ConsOld = (1 + InterestRate) * Savings(-1);

[name='Capital Market Clearing (Law of Motion)']
(1 + n) * (1 + g) * Capital = Savings;

[name='TFP']
LoggedProductivity = rho_z * LoggedProductivity(-1) + ProductivityInnovation;

end;

initval;
    Capital = k_ss;
    Output = y_ss;
    Wage = w_ss;
    InterestRate = r_ss;
    Savings = s_ss;
    ConsYoung = c1_ss;
    ConsOld = c2_ss;
    LoggedProductivity = 0;
end;

steady;
check;

shocks;
    var ProductivityInnovation; stderr @{productivity_shock_stderr};
end;

stoch_simul(order=1, periods=@{periods}, drop=0, irf=0, nomoments, nofunctions, nograph, nocorr, noprint);

% OLG Model - Diamond (1965) with Growth
% 
% Structure:
%   - 2 generations: Young (1) and Old (2)
%   - Log utility function: U = log(c1) + beta * log(c2)
%   - Variables are normalized by effective labor (Technology * Population)
%
% Timing convention (consistent with Ramsey script):
%   - Capital refers to k_t (end-of-period stock, result of savings at t)
%   - Production at t uses Capital(-1)
%
% Equations:
%   1. Euler: 1/c1_t = beta * (1 + r_{t+1}) / c2_{t+1}
%   2. Budget Young: c1_t + s_t = w_t
%   3. Budget Old:   c2_t = (1 + r_t) * s_{t-1}
%   4. Capital Market: (1+n)(1+g) * Capital_t = Savings_t
%   5. Factor prices: w_t and r_t determined by Capital(-1)

var ConsYoung       $ConsYoung$ (long_name='consumption young')
    ConsOld         $ConsOld$ (long_name='consumption old')
    Savings         $Savings$ (long_name='savings per effective worker')
    Capital         $Capital$ (long_name='capital stock (end of period)')
    Output          $Output$ (long_name='output per effective worker')
    Wage            $Wage$ (long_name='wage rate')
    InterestRate    $InterestRate$ (long_name='rental rate of capital');

parameters alpha beta delta n g start_capital k_ss y_ss w_ss r_ss s_ss c1_ss c2_ss;

alpha = @{alpha};
beta = @{beta};
delta = @{delta};
n = @{n};
g = @{g};
start_capital = @{start_capital};

k_ss = (beta * (1 - alpha) / ((1 + beta) * (1 + n) * (1 + g))) ^ (1 / (1 - alpha));
y_ss = k_ss^alpha;
w_ss = (1 - alpha) * k_ss^alpha;
r_ss = alpha * k_ss^(alpha - 1) - delta;
s_ss = beta / (1 + beta) * w_ss;
c1_ss = w_ss - s_ss;
c2_ss = (1 + r_ss) * s_ss;

model;

[name='Production (per effective labor)']
Output = Capital(-1)^alpha;

[name='Wage determination (MPL)']
Wage = (1 - alpha) * Capital(-1)^alpha;

[name='Interest Rate determination (MPK)']
InterestRate = alpha * Capital(-1)^(alpha - 1) - delta;

[name='Euler Equation (Log Utility)']
1/ConsYoung = beta * (1 + InterestRate(+1)) / ConsOld(+1);

[name='Budget Constraint - Young']
ConsYoung + Savings = Wage;

[name='Budget Constraint - Old']
ConsOld = (1 + InterestRate) * Savings(-1);

[name='Capital Market Clearing (Law of Motion)']
(1 + n) * (1 + g) * Capital = Savings;

end;

initval;
    Capital = start_capital;
    Output = Capital^alpha;
    Wage = (1 - alpha) * Capital^alpha;
    InterestRate = alpha * Capital^(alpha - 1) - delta;
    Savings = beta / (1 + beta) * Wage;
    ConsYoung = Wage - Savings;
    ConsOld = (1 + InterestRate) * Savings;
end;

endval;
    Capital = k_ss;
    Output = y_ss;
    Wage = w_ss;
    InterestRate = r_ss;
    Savings = s_ss;
    ConsYoung = c1_ss;
    ConsOld = c2_ss;
end;

perfect_foresight_setup(periods=@{periods});
perfect_foresight_solver;

% Collard (2001) - RBC Model with Correlated Shocks
% 
% Features:
%   - Technology and preference shocks with spillovers
%   - Correlated shock innovations
%   - GHH-style preferences (consumption-labor non-separable)
%
% Timing convention:
%   - Capital_t is end-of-period t capital
%   - Production at t uses Capital_{t-1}

var Output              $Output$              (long_name='Output')
    Consumption         $Consumption$         (long_name='Consumption')
    Capital             $Capital$             (long_name='Capital')
    Labor               $Labor$               (long_name='Labor')
    TechnologyShock     $TechnologyShock$     (long_name='Technology Shock')
    PreferenceShock     $PreferenceShock$     (long_name='Preference Shock');

varexo TechnologyInnovation   $TechnologyInnovation$   (long_name='Technology Shock Innovation')
       PreferenceInnovation   $PreferenceInnovation$   (long_name='Preference Shock Innovation');

parameters alpha beta delta sigma psi
           rho tau phi
           k_ss y_ss c_ss h_ss;

@#if !defined(alpha)
    @#define alpha = 0.36
@#endif

@#if !defined(beta)
    @#define beta = 0.99
@#endif

@#if !defined(delta)
    @#define delta = 0.025
@#endif

@#if !defined(psi)
    @#define psi = 0.0
@#endif

@#if !defined(rho)
    @#define rho = 0.95
@#endif

@#if !defined(tau)
    @#define tau = 0.025
@#endif

@#if !defined(phi)
    @#define phi = 0.1
@#endif

@#if !defined(h_ss_target)
    @#define h_ss_target = 0.33
@#endif

@#if !defined(technology_shock_stderr)
    @#define technology_shock_stderr = 0.009
@#endif

@#if !defined(preference_shock_stderr)
    @#define preference_shock_stderr = 0.009
@#endif

alpha = @{alpha};
beta = @{beta};
delta = @{delta};
psi = @{psi};
h_ss = @{h_ss_target};
rho = @{rho};
tau = @{tau};
phi = @{phi};

% In steady state: a = b = 0, so exp(a) = exp(b) = 1
% From Euler: 1 = beta * (alpha * y/k + 1 - delta)

% Capital-output ratio
% k_y_ratio = alpha * beta / (1 - beta * (1 - delta));

% From production: y = k^alpha * h^(1-alpha)
% From labor FOC: c * sigma * h^(1+psi) = (1-alpha) * y
% From resource constraint: y = c + delta * k

% Solving the system:
k_ss = ((alpha * beta) / (1 - beta * (1 - delta)))^(1 / (1 - alpha)) * h_ss;
y_ss = k_ss^alpha * h_ss^(1 - alpha);
c_ss = y_ss - delta * k_ss;
sigma = (1 - alpha) * y_ss / (c_ss * h_ss^(1 + psi));

model;

[name='Labor supply FOC']
Consumption * sigma * Labor^(1 + psi) = (1 - alpha) * Output;

[name='Euler equation']
Capital = beta * ((exp(PreferenceShock) * Consumption) / (exp(PreferenceShock(+1)) * Consumption(+1)))
               * (exp(PreferenceShock(+1)) * alpha * Output(+1) + (1 - delta) * Capital);

[name='Production function']
Output = exp(TechnologyShock) * Capital(-1)^alpha * Labor^(1 - alpha);

[name='Resource constraint']
Capital = exp(PreferenceShock) * (Output - Consumption) + (1 - delta) * Capital(-1);

[name='Technology shock process with spillover']
TechnologyShock = rho * TechnologyShock(-1) + tau * PreferenceShock(-1) + TechnologyInnovation;

[name='Preference shock process with spillover']
PreferenceShock = tau * TechnologyShock(-1) + rho * PreferenceShock(-1) + PreferenceInnovation;

end;

steady_state_model;
    TechnologyShock = 0;
    PreferenceShock = 0;
    Labor = h_ss;
    Capital = k_ss;
    Output = y_ss;
    Consumption = c_ss;
end;

shocks;
    var TechnologyInnovation;
    stderr @{technology_shock_stderr};
    
    var PreferenceInnovation;
    stderr @{preference_shock_stderr};
    
    var TechnologyInnovation, PreferenceInnovation = phi * @{technology_shock_stderr} * @{preference_shock_stderr};
end;

steady;
check;

stoch_simul(order=1, periods=@{periods}, drop=0, irf=0, nomoments, nofunctions, nograph, nocorr, noprint);

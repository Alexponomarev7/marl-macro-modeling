@#ifndef periods
    @#define periods = 100
@#endif

@#ifndef alpha
    @#define alpha = 0.33
@#endif
@#ifndef i_y
    @#define i_y = 0.25
@#endif
@#ifndef k_y
    @#define k_y = 10.4
@#endif
@#ifndef rho
    @#define rho = 0.97
@#endif
@#ifndef beta
    @#define beta = 1 / (alpha / k_y + (1 - delta))
@#endif
@#ifndef delta
    @#define delta = i_y / k_y
@#endif
@#ifndef l_ss
    @#define l_ss = 0.33
@#endif

var y ${y}$ (long_name='Output')
    c ${c}$ (long_name='Consumption')
    k ${k}$ (long_name='Capital')
    l ${l}$ (long_name='Hours Worked')
    z ${z}$ (long_name='Total Factor Productivity')
    invest ${i}$ (long_name='Investment');

varexo eps_z ${\varepsilon_z}$ (long_name='TFP Shock')
       eps_cap ${\varepsilon_{cap}}$ (long_name='Capital Shock');

parameters beta ${\beta}$ (long_name='Discount Factor')
           psi ${\psi}$ (long_name='Labor Disutility Parameter')
           delta ${\delta}$ (long_name='Depreciation Rate')
           alpha ${\alpha}$ (long_name='Capital Share')
           rho ${\rho}$ (long_name='Persistence TFP Shock')
           i_y ${\frac{I}{Y}}$ (long_name='Investment-Output Ratio')
           k_y ${\frac{K}{Y}}$ (long_name='Capital-Output Ratio')
           l_ss ${l_{ss}}$ (long_name='Steady State Hours Worked')
           k_ss ${k_{ss}}$ (long_name='Steady State Capital')
           i_ss ${i_{ss}}$ (long_name='Steady State Investment')
           y_ss ${y_{ss}}$ (long_name='Steady State Output')
           c_ss ${c_{ss}}$ (long_name='Steady State Consumption');

alpha = @{alpha};
i_y = @{i_y};
k_y = @{k_y};
rho = @{rho};
beta = @{beta};
delta = @{delta};
l_ss = @{l_ss};

model;
    psi * exp(c) / (1 - exp(l)) = (1 - alpha) * exp(z) * (exp(k) / exp(l))^alpha;
    1 / exp(c) = beta / exp(c(+1)) * (alpha * exp(z(+1)) * (exp(k(+1)) / exp(l(+1)))^(alpha - 1) + (1 - delta));
    exp(k) = exp(-eps_cap) * (exp(invest(-1)) + (1 - delta) * exp(k(-1)));
    exp(y) = exp(z) * exp(k)^alpha * exp(l)^(1 - alpha);
    z = rho * z(-1) + eps_z;
    exp(invest) = exp(y) - exp(c);
end;

steady_state_model;
    delta = i_y / k_y;
    beta = 1 / (alpha / k_y + (1 - delta));
    l_ss = 0.33;
    k_ss = ((1 / beta - (1 - delta)) / alpha)^(1 / (alpha - 1)) * l_ss;
    i_ss = delta * k_ss;
    y_ss = k_ss^alpha * l_ss^(1 - alpha);
    c_ss = k_ss^alpha * l_ss^(1 - alpha) - i_ss;
    psi = (1 - alpha) * (k_ss / l_ss)^alpha * (1 - l_ss) / c_ss;
    invest = log(i_ss);
    w = log((1 - alpha) * y_ss / l_ss);
    r = 4 * alpha * y_ss / k_ss;
    y = log(y_ss);
    k = log(k_ss);
    c = log(c_ss);
    l = log(l_ss);
    z = 0;
    ghat = 0;
end;

shocks;
    var eps_z = 1;
    var eps_cap = 1;
end;

steady;
check;

stoch_simul(order=1, periods=@{periods}, nomoments, nofunctions, nograph, nocorr, noprint);
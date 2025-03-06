@#ifndef periods
    @#define periods = 100
@#endif

@#ifndef sigma
    @#define sigma = 5
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
@#ifndef x
    @#define x = 0.0055
@#endif
@#ifndef n
    @#define n = 0.0027
@#endif
@#ifndef rho
    @#define rho = 0.97
@#endif
@#ifndef rhog
    @#define rhog = 0.98
@#endif
@#ifndef gshare
    @#define gshare = 0.2038
@#endif

var y ${y}$ (long_name='Output')
    c ${c}$ (long_name='Consumption')
    k ${k}$ (long_name='Capital')
    l ${l}$ (long_name='Hours Worked')
    z ${z}$ (long_name='Total Factor Productivity')
    ghat ${\hat g}$ (long_name='Government Spending')
    r ${r}$ (long_name='Annualized Interest Rate')
    w ${w}$ (long_name='Real Wage')
    invest ${i}$ (long_name='Investment');

varexo eps_z ${\varepsilon_z}$ (long_name='TFP Shock')
       eps_g ${\varepsilon_g}$ (long_name='Government Spending Shock');

parameters beta ${\beta}$ (long_name='Discount Factor')
           psi ${\psi}$ (long_name='Labor Disutility Parameter')
           sigma ${\sigma}$ (long_name='Risk Aversion')
           delta ${\delta}$ (long_name='Depreciation Rate')
           alpha ${\alpha}$ (long_name='Capital Share')
           rho ${\rho}$ (long_name='Persistence TFP Shock')
           gammax ${\gamma_x}$ (long_name='Composite Growth Rate')
           rhog ${\rho_g}$ (long_name='Persistence Government Spending Shock')
           gshare ${\frac{G}{Y}}$ (long_name='Government Spending Share')
           l_ss ${l_{ss}}$ (long_name='Steady State Hours Worked')
           k_ss ${k_{ss}}$ (long_name='Steady State Capital')
           i_ss ${i_{ss}}$ (long_name='Steady State Investment')
           y_ss ${y_{ss}}$ (long_name='Steady State Output')
           g_ss ${\bar G}$ (long_name='Steady State Government Spending')
           c_ss ${c_{ss}}$ (long_name='Steady State Consumption')
           n ${n}$ (long_name='Population Growth')
           x ${x}$ (long_name='Technology Growth')
           k_y ${\frac{K}{Y}}$ (long_name='Capital-Output Ratio')
           i_y ${\frac{I}{Y}}$ (long_name='Investment-Output Ratio');

sigma = @{sigma};
alpha = @{alpha};
i_y = @{i_y};
k_y = @{k_y};
x = @{x};
n = @{n};
rho = @{rho};
rhog = @{rhog};
gshare = @{gshare};

model;
    psi * exp(c)^sigma * 1 / (1 - exp(l)) = (1 - alpha) * exp(z) * (exp(k(-1)) / exp(l))^alpha;
    exp(c)^(-sigma) = beta / gammax * exp(c(+1))^(-sigma) * (alpha * exp(z(+1)) * (exp(k) / exp(l(+1)))^(alpha - 1) + (1 - delta));
    gammax * exp(k) = exp(y) - exp(c) + (1 - delta) * exp(k(-1)) - g_ss * exp(ghat);
    exp(y) = exp(z) * exp(k(-1))^alpha * exp(l)^(1 - alpha);
    z = rho * z(-1) + eps_z;
    ghat = rhog * ghat(-1) + eps_g;
    exp(w) = (1 - alpha) * exp(y) / exp(l);
    r = 4 * alpha * exp(y) / exp(k(-1));
    exp(invest) = exp(y) - exp(c) - g_ss * exp(ghat);
end;

steady_state_model;
    gammax = (1 + n) * (1 + x);
    delta = i_y / k_y - x - n - n * x;
    beta = (1 + x) * (1 + n) / (alpha / k_y + (1 - delta));
    l_ss = 0.33;
    k_ss = ((1 / beta * (1 + n) * (1 + x) - (1 - delta)) / alpha)^(1 / (alpha - 1)) * l_ss;
    i_ss = (x + n + delta + n * x) * k_ss;
    y_ss = k_ss^alpha * l_ss^(1 - alpha);
    g_ss = gshare * y_ss;
    c_ss = (1 - gshare) * k_ss^alpha * l_ss^(1 - alpha) - i_ss;
    psi = (1 - alpha) * (k_ss / l_ss)^alpha * (1 - l_ss) / c_ss^sigma;
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
    var eps_z = 0.0068^2;
    var eps_g = 0.0105^2;
end;

steady;
check;

stoch_simul(order=2, irf=0, periods=@{periods}, nomoments, nofunctions, nograph, nocorr, noprint);
@#ifndef periods
    @#define periods = 100
@#endif

var y ${y}$ (long_name='Output')
    c ${c}$ (long_name='Consumption')
    k ${k}$ (long_name='Capital')
    l ${l}$ (long_name='Hours Worked')
    z ${z}$ (long_name='Total Factor Productivity')
    r ${r}$ (long_name='Annualized Interest Rate')
    w ${w}$ (long_name='Real Wage')
    invest ${i}$ (long_name='Investment');

varexo eps_z_news ${\varepsilon_z^{news}}$ (long_name='Anticipated TFP Shock')
       eps_z_surprise ${\varepsilon_z^{surprise}}$ (long_name='Unanticipated TFP Shock');

parameters beta ${\beta}$ (long_name='Discount Factor')
           psi ${\psi}$ (long_name='Labor Disutility Parameter')
           sigma ${\sigma}$ (long_name='Risk Aversion')
           delta ${\delta}$ (long_name='Depreciation Rate')
           alpha ${\alpha}$ (long_name='Capital Share')
           rhoz ${\rho_z}$ (long_name='Persistence TFP Shock')
           gammax ${\gamma_x}$ (long_name='Composite Growth Rate')
           n ${n}$ (long_name='Population Growth')
           x ${x}$ (long_name='Technology Growth')
           i_y ${\frac{I}{Y}}$ (long_name='Investment-Output Ratio')
           k_y ${\frac{K}{Y}}$ (long_name='Capital-Output Ratio');

sigma = 1;
alpha = 0.33;
i_y = 0.25;
k_y = 10.4;
x = 0.0055;
n = 0.0027;
rhoz = 0.97;

model;
    exp(c)^(-sigma) = beta / gammax * exp(c(+1))^(-sigma) * (alpha * exp(z(+1)) * (exp(k) / exp(l(+1)))^(alpha - 1) + (1 - delta));
    psi * exp(c)^sigma * 1 / (1 - exp(l)) = exp(w);
    gammax * exp(k) = (1 - delta) * exp(k(-1)) + exp(invest);
    exp(y) = exp(invest) + exp(c);
    exp(y) = exp(z) * exp(k(-1))^alpha * exp(l)^(1 - alpha);
    exp(w) = (1 - alpha) * exp(y) / exp(l);
    r = 4 * alpha * exp(y) / exp(k(-1));
    z = rhoz * z(-1) + eps_z_surprise + eps_z_news(-8);
end;

steady_state_model;
    gammax = (1 + n) * (1 + x);
    delta = i_y / k_y - x - n - n * x;
    beta = gammax / (alpha / k_y + (1 - delta));
    l_ss = 0.33;
    k_ss = ((1 / beta * gammax - (1 - delta)) / alpha)^(1 / (alpha - 1)) * l_ss;
    i_ss = (x + n + delta + n * x) * k_ss;
    y_ss = k_ss^alpha * l_ss^(1 - alpha);
    c_ss = y_ss - i_ss;
    psi = (1 - alpha) * (k_ss / l_ss)^alpha * (1 - l_ss) / c_ss^sigma;
    invest = log(i_ss);
    w = log((1 - alpha) * y_ss / l_ss);
    r = 4 * alpha * y_ss / k_ss;
    y = log(y_ss);
    k = log(k_ss);
    c = log(c_ss);
    l = log(l_ss);
    z = 0;
end;

shocks;
    var eps_z_news = 1;
    var eps_z_surprise = 1;
end;

steady;
check;

stoch_simul(order=1, irf=40, periods=@{periods}, nomoments, nofunctions, nograph, nocorr, noprint);
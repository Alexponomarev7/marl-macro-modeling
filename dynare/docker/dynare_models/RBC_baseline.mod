@#ifndef periods
    @#define periods = 100
@#endif

var y           ${y}$         (long_name='Output')
    c           ${c}$         (long_name='Consumption')
    k           ${k}$         (long_name='Capital')
    l           ${l}$         (long_name='Hours Worked')
    z           ${z}$         (long_name='Total Factor Productivity')
    ghat        ${\hat g}$    (long_name='Government Spending')
    r           ${r}$         (long_name='Annualized Interest Rate')
    w           ${w}$         (long_name='Real Wage')
    invest      ${i}$         (long_name='Investment')
    log_y       ${\log(y)}$   (long_name='Log Output')
    log_k       ${\log(k)}$   (long_name='Log Capital Stock')
    log_c       ${\log(c)}$   (long_name='Log Consumption')
    log_l       ${\log(l)}$   (long_name='Log Labor')
    log_w       ${\log(w)}$   (long_name='Log Real Wage')
    log_invest  ${\log(i)}$   (long_name='Log Investment');

varexo eps_z ${\varepsilon_z}$ (long_name='TFP Shock')
       eps_g ${\varepsilon_g}$ (long_name='Government Spending Shock');

parameters beta    ${\beta}$   (long_name='Discount Factor')
           psi     ${\psi}$    (long_name='Labor Disutility Parameter')
           sigma   ${\sigma}$  (long_name='Risk Aversion')
           delta   ${\delta}$  (long_name='Depreciation Rate')
           alpha   ${\alpha}$  (long_name='Capital Share')
           rhoz    ${\rho_z}$  (long_name='Persistence TFP Shock')
           rhog    ${\rho_g}$  (long_name='Persistence Government Spending Shock')
           gammax  ${\gamma_x}$ (long_name='Composite Growth Rate')
           gshare  ${\frac{G}{Y}}$ (long_name='Government Spending Share')
           n       ${n}$       (long_name='Population Growth')
           x       ${x}$       (long_name='Technology Growth')
           i_y     ${\frac{I}{Y}}$ (long_name='Investment-Output Ratio')
           k_y     ${\frac{K}{Y}}$ (long_name='Capital-Output Ratio')
           g_ss    ${\bar G}$  (long_name='Steady State Government Spending');

sigma = 1;
alpha = 0.33;
i_y = 0.25;
k_y = 10.4;
x = 0.0055;
n = 0.0027;
rhoz = 0.97;
rhog = 0.989;
gshare = 0.2038;

model;
    c^(-sigma) = beta / gammax * c(+1)^(-sigma) * (alpha * exp(z(+1)) * (k / l(+1))^(alpha - 1) + (1 - delta));
    psi * c^sigma * 1 / (1 - l) = w;
    gammax * k = (1 - delta) * k(-1) + invest;
    y = invest + c + g_ss * exp(ghat);
    y = exp(z) * k(-1)^alpha * l^(1 - alpha);
    w = (1 - alpha) * y / l;
    r = 4 * alpha * y / k(-1);
    z = rhoz * z(-1) + eps_z;
    ghat = rhog * ghat(-1) + eps_g;
    log_y = log(y);
    log_k = log(k);
    log_c = log(c);
    log_l = log(l);
    log_w = log(w);
    log_invest = log(invest);
end;

steady_state_model;
    gammax = (1 + n) * (1 + x);
    delta = i_y / k_y - x - n - n * x;
    beta = (1 + x) * (1 + n) / (alpha / k_y + (1 - delta));
    l = 0.33;
    k = ((1 / beta * (1 + n) * (1 + x) - (1 - delta)) / alpha)^(1 / (alpha - 1)) * l;
    invest = (x + n + delta + n * x) * k;
    y = k^alpha * l^(1 - alpha);
    g = gshare * y;
    g_ss = g;
    c = (1 - gshare) * k^alpha * l^(1 - alpha) - invest;
    psi = (1 - alpha) * (k / l)^alpha * (1 - l) / c^sigma;
    w = (1 - alpha) * y / l;
    r = 4 * alpha * y / k;
    log_y = log(y);
    log_k = log(k);
    log_c = log(c);
    log_l = log(l);
    log_w = log(w);
    log_invest = log(invest);
    z = 0;
    ghat = 0;
end;

shocks;
    var eps_z = 0.66^2;
    var eps_g = 1.04^2;
end;

steady;
check;

stoch_simul(order=1, irf=40, hp_filter=1600, periods=@{periods}, nomoments, nofunctions, nograph, nocorr, noprint);
@#ifndef periods
    @#define periods = 100
@#endif

@#ifndef R
    @#define R = 1.2
@#endif
@#ifndef sigma_w
    @#define sigma_w = 1
@#endif

var y $y$                  (long_name='Output')
    c $c$                  (long_name='Consumption')
    y_m_c ${y - c}$        (long_name='Output Minus Consumption');
varexo w $w$              (long_name='Exogenous Shock');
parameters R $R$          (long_name='Parameter R')
           sigma_w ${\sigma_w}$ (long_name='Shock Scale Parameter');

sigma_w = @{sigma_w};
R = @{R};

model;
    c = c(-1) + sigma_w * (1 - R^(-1)) * w;
    y_m_c = -c(-1) + sigma_w * R^(-1) * w;
    y_m_c = y - c;
end;

steady_state_model;
    c = 0;
    y = 0;
    y_m_c = 0;
end;

shocks;
    var w = 1;
end;

steady;
check;
varobs y_m_c;
stoch_simul(order=1, periods=@{periods}, nomoments, nofunctions, nograph, nocorr, noprint);
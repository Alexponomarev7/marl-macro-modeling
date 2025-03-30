var y                   (long_name='Output')
    c                   (long_name='Consumption')
    y_m_c               (long_name='Output Minus Consumption');
varexo w                (long_name='Exogenous Shock');
parameters R            (long_name='Parameter R')
           sigma_w      (long_name='Shock Scale Parameter');

sigma_w = {sigma_w};
R = {R};

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
    var w; periods {shock_periods}; values {shock_values};
end;

steady;
check;

perfect_foresight_setup(periods={periods});
perfect_foresight_solver;
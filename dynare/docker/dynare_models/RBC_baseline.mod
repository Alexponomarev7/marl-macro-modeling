var Consumption, Capital, LoggedProductivity;

varexo LoggedProductivityInnovation;

parameters beta, alpha, delta, rho;

beta = .985;
alpha = 1/3;
delta = alpha/10;
rho = .9;

model;

[name='Euler equation']
1/Consumption = beta/Consumption(1)*(alpha*exp(LoggedProductivity(1))*Capital^(alpha-1)+1-delta);

[name='Physical capital stock law of motion']
Capital = exp(LoggedProductivity)*Capital(-1)^alpha+(1-delta)*Capital(-1)-Consumption;

[name='Logged productivity law of motion']
LoggedProductivity = rho*LoggedProductivity(-1)+LoggedProductivityInnovation;

end;

steady;

initval;
  LoggedProductivityInnovation = 0;
  LoggedProductivity = .05;
  Capital = 0.5;
end;

endval;
  LoggedProductivity = LoggedProductivityInnovation/(1-rho);
  Capital = (exp(LoggedProductivity)*alpha/(1/beta-1+delta))^(1/(1-alpha));
  Consumption = exp(LoggedProductivity)*Capital^alpha-delta*Capital;
end;

@{shocks}

perfect_foresight_setup(periods=@{periods});
perfect_foresight_solver;

var y pi eps theta i;

varexo ea eg;

parameters sigma beta kappa phi_pi phi_y rho_eps rho_theta;

% set parameters
sigma = 1.0;
beta = 0.99;
kappa = 0.01;
phi_pi = 1.5;
phi_y = 0.5 / 4;
rho_eps = 0.9;
rho_theta = 0.9;


model(linear);

y = y(+1) - 1/sigma * (i - pi(+1)) + eps;

pi = kappa*y + beta * pi(+1);

i = phi_pi * pi + phi_y * y + theta;

eps = rho_eps * eps(-1) + ea;

theta = rho_theta * theta(-1) + eg;

end;

shocks;
var ea = 1;
var eg = 1;
end;

stoch_simul(order=1,irf=20,nograph,ar=1,periods=100);
var C ${C}$ (long_name='Consumption')
    W_real ${\frac{W}{P}}$ (long_name='Real Wage')
    Pi ${\Pi}$ (long_name='Inflation')
    A  ${A}$   (long_name='AR(1) Technology Process')
    N ${N}$   (long_name='Hours Worked')
    R ${R^n}$    (long_name='Nominal Interest Rate')
    realinterest ${R^{r}}$ (long_name='Real Interest Rate')
    Y ${Y}$ (long_name='Output')
    m_growth_ann ${\Delta M}$ (long_name='Money Growth');
varexo eps_A ${\varepsilon_A}$   (long_name='Technology Shock')
       eps_m ${\varepsilon_m}$ (long_name='Monetary Policy Shock')
       ;

parameters alppha ${\alpha}$ (long_name='capital share')
    betta ${\beta}$ (long_name='discount factor')
    rho ${\rho}$ (long_name='autocorrelation technology shock')
    siggma ${\sigma}$ (long_name='log utility')
    phi ${\phi}$ (long_name='unitary Frisch elasticity')
    phi_pi ${\phi_{\pi}}$ (long_name='inflation feedback Taylor Rule')
    eta ${\eta}$ (long_name='semi-elasticity of money demand')
    ;

%----------------------------------------------------------------
% Follows parametrization of Chapter 3, p. 52
%----------------------------------------------------------------

alppha=0.33;
betta=0.99;
rho=0.9;
siggma=1;
phi=1;
phi_pi=1.5;
eta  =4;


%----------------------------------------------------------------
% First Order Conditions
%----------------------------------------------------------------

model;
//1. FOC Wages, eq. (6)
W_real=C^siggma*N^phi;
//2. Euler equation eq. (7)
1/R=betta*(C(+1)/C)^(-siggma)/Pi(+1);
//3. Production function eq. (8)
A*N^(1-alppha)= C;
//4. FOC wages firm, eq. (13)
W_real=(1-alppha)*A*N^(-alppha);
//5. Definition Real interest rate
realinterest=R/Pi(+1);
//6. Monetary Policy Rule, eq. (22)
R=1/betta*Pi^phi_pi+eps_m;
//7. Market Clearing, eq. (15)
C=Y;
//8. Technology Shock
log(A)=rho*log(A(-1))+eps_A;
//11. Money growth (derived from eq. (10))
m_growth_ann=4*(log(Y)-log(Y(-1))-eta*(log(R)-log(R(-1)))+log(Pi));

end;

%----------------------------------------------------------------
%  define shock variances
%---------------------------------------------------------------

shocks;
var eps_A; stderr 1;
var eps_m; stderr 1;
end;

%----------------------------------------------------------------
%  Initial Values for steady state
%---------------------------------------------------------------

steady_state_model;
A=1;
R=1/betta;
Pi=1;
realinterest=R;
N=(1-alppha)^(1/((1-siggma)*alppha+phi+siggma));
C=A*N^(1-alppha);
W_real=(1-alppha)*A*N^(-alppha);
Y=C;
m_growth_ann=0;
end;

steady;
check;

stoch_simul(order=1,nograph,ar=1,periods=100);
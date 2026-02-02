% Faia & Monacelli (2008): Optimal Monetary Policy in a Small Open Economy with Home Bias
%
% Key features:
%   - Small open economy (takes foreign variables as given)
%   - Home bias in consumption (gamma_h = share of home goods)
%   - CES aggregator over home and foreign goods
%   - Rotemberg price adjustment costs
%   - Complete international asset markets (risk sharing)
%   - Deviations from PPP due to home bias
%   - Taylor rule monetary policy

var Consumption             $Consumption$             (long_name='aggregate consumption')
    ConsumptionH            $ConsumptionH$            (long_name='consumption of home goods')
    ConsumptionF            $ConsumptionF$            (long_name='consumption of foreign goods')
    Exports                 $Exports$                 (long_name='exports')
    Labor                   $Labor$                   (long_name='labor supply')
    Output                  $Output$                  (long_name='domestic output')
    InflationH              $InflationH$              (long_name='domestic goods inflation')
    InflationCPI            $InflationCPI$            (long_name='CPI inflation')
    NominalInterestRate     $NominalInterestRate$     (long_name='nominal interest rate')
    RealInterestRate        $RealInterestRate$        (long_name='real interest rate')
    TermsOfTrade            $TermsOfTrade$            (long_name='terms of trade PF over PH')
    RealExchangeRate        $RealExchangeRate$        (long_name='real exchange rate')
    NominalExchangeRate     $NominalExchangeRate$     (long_name='nominal exchange rate level')
    DepreciationRate        $DepreciationRate$        (long_name='nominal depreciation rate')
    MarginalCost            $MarginalCost$            (long_name='real marginal cost')
    Wage                    $Wage$                    (long_name='real wage in CPI units')
    Lambda                  $Lambda$                  (long_name='marginal utility of consumption')
    Productivity            $Productivity$            (long_name='productivity level')
    LogProductivity         $LogProductivity$         (long_name='log productivity')
    OutputForeign           $OutputForeign$           (long_name='foreign output')
    ConsumptionForeign      $ConsumptionForeign$      (long_name='foreign consumption')
    InflationForeign        $InflationForeign$        (long_name='foreign CPI inflation')
    PriceRatioH             $PriceRatioH$             (long_name='relative price PH over P')
    LogOutput               $LogOutput$               (long_name='log output')
    LogConsumption          $LogConsumption$          (long_name='log consumption')
    LogRealExchangeRate     $LogRealExchangeRate$     (long_name='log real exchange rate')
    LogTermsOfTrade         $LogTermsOfTrade$         (long_name='log terms of trade')
    LogInflationH           $LogInflationH$           (long_name='log domestic inflation');

varexo ProductivityInnovation     $ProductivityInnovation$     (long_name='productivity shock')
       ForeignOutputInnovation    $ForeignOutputInnovation$    (long_name='foreign output shock')
       ForeignInflationInnovation $ForeignInflationInnovation$ (long_name='foreign inflation shock');
 
parameters sigma           ${\sigma}$          (long_name='risk aversion')
           phi             ${\phi}$            (long_name='inverse Frisch elasticity')
           eta             ${\eta}$            (long_name='elasticity of substitution H-F goods')
           gamma_h         ${\gamma_h}$        (long_name='home bias degree')
           beta            ${\beta}$           (long_name='discount factor')
           epsilon         ${\epsilon}$        (long_name='elasticity across varieties')
           psi_p           ${\psi}$            (long_name='Rotemberg adjustment cost')
           rho_a           ${\rho_a}$          (long_name='productivity persistence')
           rho_ystar       ${\rho_{y^*}}$      (long_name='foreign output persistence')
           rho_pistar      ${\rho_{\pi^*}}$    (long_name='foreign inflation persistence')
           phi_pi          ${\phi_\pi}$        (long_name='Taylor rule inflation response')
           phi_y           ${\phi_y}$          (long_name='Taylor rule output response')
           omega           ${\omega}$          (long_name='openness degree')
           r_f             ${r_f}$             (long_name='foreign interest rate')
           kappa_L         ${\kappa_L}$        (long_name='labor disutility parameter')
           NominalInterestRate_ss Lambda_ss Consumption_ss ConsumptionH_ss ConsumptionF_ss
           Labor_ss Output_ss Wage_ss MarginalCost_ss PriceRatioH_ss OutputForeign_ss;

beta = @{beta};
sigma = @{sigma};
phi = @{phi};
eta = @{eta};
gamma_h = @{gamma_h};
epsilon = @{epsilon};
psi_p = @{psi_p};
rho_a = @{rho_a};
rho_ystar = @{rho_ystar};
rho_pistar = @{rho_pistar};
phi_pi = @{phi_pi};
phi_y = @{phi_y};

omega = 1 - gamma_h;
r_f = 1 / beta;

NominalInterestRate_ss = 1 / beta;
MarginalCost_ss = (epsilon - 1) / epsilon;
PriceRatioH_ss = 1;
OutputForeign_ss = 1;
Consumption_ss = OutputForeign_ss; 
Output_ss = 1;
Labor_ss = 1;
kappa_L = MarginalCost_ss / (Consumption_ss^sigma * Labor_ss^phi);
Wage_ss = MarginalCost_ss;
ConsumptionH_ss = gamma_h * Consumption_ss;
ConsumptionF_ss = (1 - gamma_h) * Consumption_ss;
Lambda_ss = Consumption_ss^(-sigma);

model;

[name='Marginal utility of consumption']
Lambda = Consumption^(-sigma);

[name='Labor supply FOC']
Wage = kappa_L * Consumption^sigma * Labor^phi;

[name='CES demand for home goods']
ConsumptionH = gamma_h * PriceRatioH^(-eta) * Consumption;

[name='CES demand for foreign goods']
ConsumptionF = (1 - gamma_h) * (TermsOfTrade * PriceRatioH)^eta * PriceRatioH^(-eta) * Consumption;

[name='Exports demand']
Exports = (1 - gamma_h) * TermsOfTrade^eta * OutputForeign;

[name='Goods Market Clearing']
Output = ConsumptionH + Exports;

[name='Price index (PH/P relationship)']
1 = gamma_h * PriceRatioH^(1-eta) + (1 - gamma_h) * (TermsOfTrade * PriceRatioH)^(1-eta);

[name='CPI inflation']
InflationCPI = InflationH * PriceRatioH / PriceRatioH(-1);

[name='Terms of trade dynamics']
TermsOfTrade = TermsOfTrade(-1) * InflationForeign * DepreciationRate / InflationH;

[name='Real exchange rate']
RealExchangeRate = TermsOfTrade^(1 - gamma_h);

[name='Nominal exchange rate dynamics']
NominalExchangeRate = NominalExchangeRate(-1) * DepreciationRate;

[name='Production function']
Output = Productivity * Labor;

[name='Marginal cost']
MarginalCost = Wage / (Productivity * PriceRatioH);

[name='Phillips curve (Rotemberg)']
psi_p * (InflationH - 1) * InflationH = psi_p * beta * Lambda(+1) / Lambda * (InflationH(+1) - 1) * InflationH(+1) * Output(+1) / Output + (1 - epsilon) + epsilon * MarginalCost;

[name='International risk sharing']
Consumption = ConsumptionForeign * RealExchangeRate^(1/sigma);

[name='UIP condition']
DepreciationRate(+1) = NominalInterestRate / r_f;

[name='Real interest rate']
RealInterestRate = NominalInterestRate / InflationCPI(+1);

[name='Taylor rule']
log(NominalInterestRate / steady_state(NominalInterestRate)) = phi_pi * log(InflationCPI) + phi_y * log(Output / steady_state(Output));

[name='Productivity process']
LogProductivity = rho_a * LogProductivity(-1) + ProductivityInnovation;

[name='Productivity level']
Productivity = exp(LogProductivity);

[name='Foreign output process']
log(OutputForeign) = rho_ystar * log(OutputForeign(-1)) + ForeignOutputInnovation;

[name='Foreign consumption']
ConsumptionForeign = OutputForeign;

[name='Foreign inflation process']
log(InflationForeign) = rho_pistar * log(InflationForeign(-1)) + ForeignInflationInnovation;

[name='Log output']
LogOutput = log(Output);

[name='Log consumption']
LogConsumption = log(Consumption);

[name='Log real exchange rate']
LogRealExchangeRate = log(RealExchangeRate);

[name='Log terms of trade']
LogTermsOfTrade = log(TermsOfTrade);

[name='Log domestic inflation']
LogInflationH = log(InflationH);

end;

steady_state_model;
    LogProductivity = 0;
    Productivity = 1;
    OutputForeign = OutputForeign_ss;
    ConsumptionForeign = OutputForeign_ss;
    InflationForeign = 1;
    InflationH = 1;
    InflationCPI = 1;
    TermsOfTrade = 1;
    RealExchangeRate = 1;
    NominalExchangeRate = 1;
    DepreciationRate = 1;
    PriceRatioH = PriceRatioH_ss;
    NominalInterestRate = NominalInterestRate_ss;
    RealInterestRate = NominalInterestRate_ss;
    Consumption = ConsumptionForeign;
    ConsumptionH = ConsumptionH_ss;
    ConsumptionF = ConsumptionF_ss;
    Exports = (1 - gamma_h) * TermsOfTrade^eta * OutputForeign;
    MarginalCost = MarginalCost_ss;
    Wage = Wage_ss;
    Labor = Labor_ss;
    Output = Output_ss;
    Lambda = Lambda_ss;
    LogOutput = log(Output);
    LogConsumption = log(Consumption);
    LogRealExchangeRate = 0;
    LogTermsOfTrade = 0;
    LogInflationH = 0;
end;

steady;
check;

shocks;
    var ProductivityInnovation;
    stderr @{stderr_a};
    var ForeignOutputInnovation;
    stderr @{stderr_ystar};
    var ForeignInflationInnovation;
    stderr @{stderr_pistar};
end;

stoch_simul(order=1, periods=@{periods}, irf=0, nomoments, nofunctions, nograph, nocorr, noprint);

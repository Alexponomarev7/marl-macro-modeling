var C       ${C}$ (long_name='consumption')
    K       ${K}$ (long_name='capital')
    Y       ${Y}$ (long_name='output')
    invest  ${I}$ (long_name='investment');

parameters k_ss c_ss alpha beta delta start_capital;

@#if !defined(alpha)
  @#define alpha = 0.33
@#endif

@#if !defined(beta)
  @#define beta = 0.96
@#endif

@#if !defined(delta)
  @#define delta = 0.1
@#endif

@#if !defined(start_capital)
  @#define start_capital = 1
@#endif

alpha = @{alpha};
beta = @{beta};
delta = @{delta};
start_capital = @{start_capital};
periods = @{periods};

k_ss = ((1/beta - (1 - delta))/alpha)^(1/(alpha - 1));
c_ss = k_ss^alpha - delta*k_ss;

model;
[name='Law of motion capital']
K=(1-delta)*K(-1)+invest;
[name='resource constraint']
invest+C=Y;
[name='behavioral rule savings']
1/C=beta*1/C(+1)*(alpha*Y(+1)/K+(1-delta));
[name='production function']
Y=K(-1)^alpha;
end;

initval;
    K = start_capital;
end;

endval;
    K = k_ss;
    C = c_ss;
    Y = K^alpha;
    invest = Y - C;
end;

perfect_foresight_setup(periods=@{periods});
perfect_foresight_solver;

% for debugging
% rplot K;
% rplot C;

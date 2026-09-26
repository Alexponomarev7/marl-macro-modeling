function oo_ = mit_shock_solver(M_, options_, oo_)
% Perfect foresight with unanticipated ("MIT") shocks: each shock becomes known only in the period
% it is dated, and agents then re-plan from the realized state, expecting no further shocks.
% Equivalent to perfect_foresight_with_expectation_errors with learnt_in = each shock's period.
%
% Usage in a .mod file: declare all shocks in a plain `shocks;` block, then
%     perfect_foresight_setup(periods=...);
%     oo_ = mit_shock_solver(M_, options_, oo_);

periods = options_.periods;
L = M_.maximum_lag;
F = M_.maximum_lead;
exo_realized = oo_.exo_simul;   % (L + periods + F) x exo_nbr
endo = oo_.endo_simul;          % initial conditions | guess path | terminal values

deviates = exo_realized(L+1:L+periods, :) ~= repmat(oo_.exo_steady_state', periods, 1);
info_periods = unique([1, find(any(deviates, 2))']);

for p = info_periods
    n = periods - p + 1;
    oo_.endo_simul = endo(:, p:end);                     % first L columns: realized past
    exo = repmat(oo_.exo_steady_state', L + n + F, 1);
    exo(1:L, :) = exo_realized(p:p+L-1, :);
    exo(L+1, :) = exo_realized(L+p, :);                 % shocks dated p are known; later ones are not
    oo_.exo_simul = exo;
    options_.periods = n;
    oo_ = perfect_foresight_solver(M_, options_, oo_);
    if ~oo_.deterministic_simulation.status
        error('mit_shock_solver: no solution for information available at period %d', p);
    end
    endo(:, p:end) = oo_.endo_simul;
end

oo_.endo_simul = endo;
oo_.exo_simul = exo_realized;
end

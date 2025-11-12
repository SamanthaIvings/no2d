function [TT]=BPR_density_smooth(time,x,criticalDensity,params)
% introduce small perturbation eps to ensure smooth travel time growth
% at low densities - prevents the system behaving like an all-or-nothing
% assignment in the early stages

alpha = params(:,1); beta = params(:,2); eps = unique(params(:,3));

ratio = x ./ criticalDensity; % compute the density ratio
ratio = max(ratio, eps); % apply epsilon only where needed

TT = time.*(1+alpha.*ratio.^beta);

end
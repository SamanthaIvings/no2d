function [TT]=BPR_density(time,x,criticalDensity,params)

alpha = params(1); beta = params(2);

TT= time.*(1+alpha.*(x./criticalDensity).^beta);

end
function [T] = BeckmannMin_UE(L,density,density_y,time,params,criticalDensity)

    alpha = params(:,1); beta = params(:,2); eps = unique(params(:,3));

    x = density+L.*(density_y-density);

    T1 = zeros(length(x),1);
    for ii = 1:length(T1)
        if x(ii)/criticalDensity(ii) >= eps
            T1(ii) = time(ii) * x(ii) * (1 + ((alpha(ii)/(beta(ii)+1))*...
                x(ii)^(beta(ii))*(criticalDensity(ii)^beta(ii))));
        else
            T1(ii) = time(ii) * x(ii) * (1 + alpha(ii)*eps^(beta(ii)));
        end
    end

    %T1 = time .* x .* (1 + ((alpha./(beta+1)).*x.^(beta).*(criticalDensity.^beta)));
    T = sum(T1);

end

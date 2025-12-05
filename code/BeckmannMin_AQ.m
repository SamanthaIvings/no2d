function [T] = BeckmannMin_AQ(L,density,density_y,params)
    %a = params(1);
    %b = params(2);
    m = params(1);

    x = density+L.*(density_y-density);

    T1 = m.*x.^2;

    %T1 = a.*x + (2*a^2/b).*exp(-b*x) - (a^2/2*b).*exp(-2*b*x);
    %T1 = a^2.*x + (2*a^2/b).*exp(-b*x) - (a^2/2*b).*exp(-2*b*x);

    T=sum(T1);
end

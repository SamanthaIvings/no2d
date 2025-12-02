function [out]=NO2_Function(x, params)
    %a = params(1);
    %b = params(2);
    m = params(1);
    
    %out = a .* (1 - exp(-b.*x));
    %out = a^2 - 2*a^2*exp(-b*x) + a^2*exp(-2*b*x);
    out = m*x;
end
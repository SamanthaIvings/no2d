function [T] = BeckmannFuncStepMin_Density(L,density,density_y,time,alpha,beta,criticalDensity)
    T1=zeros(size(time));
    for i=1:length(T1)
        T1(i)=time(i) * ...
            (density(i)+L*(density_y(i)-density(i)) + ...
            (alpha(i)/((beta(i)+1) * criticalDensity(i)^(beta(i)))) * ...
            ((density(i)+L*(density_y(i)-density(i)))^(beta(i)+1)));
    end
    T=sum(T1);
end

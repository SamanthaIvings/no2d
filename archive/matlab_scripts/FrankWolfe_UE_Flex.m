function [UEflows,crit1,crit2,L,Xa_Gi,critLog,critBests,UEflowsBest,crit1Best,crit2Best,iter,LBD,LBDBest]=FrankWolfe_UE_Flex(demand,time,topo_graph,OD_list,eps,capacity,criticalDensity,params,stepbreak,txtName,critLogName,critBestsName)

%Algorithm written according to Patriksson (2015)

%Step 0- Initialisation

topo_graph_input=topo_graph;

flow0=zeros(size(topo_graph.Edges,1),1);
density0 = flow0 .* criticalDensity./capacity;

traveltime0=BPR_density_smooth(time,density0,criticalDensity,params);

topo_graph_input.Edges.Weight=traveltime0;

% calculate the shortest path for each OD pair to assign all the demand to.

flow=flow0;

for i=1:height(topo_graph_input.Nodes)
    [~,~,E] = shortestpathtree(topo_graph_input,i,'OutputForm','cell');
    E_store{i}=E;
end

for i=1:length(OD_list)
    s=OD_list(i,3);
    t=OD_list(i,4);
    E1=E_store{s};
    edgepath=E1{t};
    flow(edgepath,1)=flow(edgepath,1)+demand(i);
end

LBD=0;
L=0;

crit1=inf;
crit2=inf;

crit1Best=crit1;
crit2Best=crit2;
UEflowsBest=flow;
iter=0;

%======Step 1 - Solve linear programming problem/ search
%direction=======
density = flow .* criticalDensity./capacity;
traveltime=BPR_density_smooth(time,density,criticalDensity,params);

% traveltime=real(traveltime);
topo_graph_input.Edges.Weight=traveltime;

while abs(crit1)>eps || abs(crit2)>eps
    L=L+1;
    
    if L==stepbreak
        disp(['Max iterations reached...',num2str(L)])
    
    
        Xa_Gi=zeros(length(flow),length(OD_list));
        for i=1:height(topo_graph_input.Nodes)
            [~,~,E] = shortestpathtree(topo_graph_input,i,'OutputForm','cell');
            E_store{i}=E;
        end
    
        for i=1:length(OD_list)
            s=OD_list(i,3);
            t=OD_list(i,4);
            E1=E_store{s};
            edgepath=E1{t};
            Xa_Gi(edgepath,i)=1;
        end
    
        UEflows=flow;
        % crit1=crit1Best;
        % crit2=crit2Best;
        break
    end
    
    % calculate the shortest path for each OD pair to assign all the demand to.
    flow_y=zeros(length(flow),1);
    
    Xa_Gi=zeros(length(flow),length(OD_list));
    
    for i=1:height(topo_graph_input.Nodes)
        [~,~,E] = shortestpathtree(topo_graph_input,i,'OutputForm','cell');
        E_store{i}=E;
    end
    
    for i=1:length(OD_list)
        s=OD_list(i,3);
        t=OD_list(i,4);
        E1=E_store{s};
        edgepath=E1{t};
        
        flow_y(edgepath,1)=flow_y(edgepath,1)+demand(i);
        Xa_Gi(edgepath,i)=1;
    end
    
    
    flow_p=flow_y-flow;
    density = flow .* criticalDensity./capacity;
    density_y = flow_y .* criticalDensity./capacity;
    density_p = density_y - density;
    %==================== step 2: Convergence Check===================
    
    %gradT=BPR_func(time,flow,capacity,alpha,beta);
    %T=flow'*(traveltime);
    %T_Bar=T+gradT'*flow_p;
    gradT = BPR_density_smooth(time,density,criticalDensity,params);
    T=flow'*(traveltime);
    T_Bar=T+gradT'*flow_p;
    
    LBD=max([LBD,T_Bar]);
    
    crit1=abs(T-LBD)/LBD;
    
    if (crit1 <= crit1Best) && (crit2 <= crit2Best)
        crit1Best = crit1;
        crit2Best = crit2;
        UEflowsBest = flow;
        iter = L;
        LBDBest = LBD;
        critBests = readmatrix(critBestsName);
        critBests(size(critBests,1)+1,:) = [crit1Best,crit2Best,iter];
        writematrix(critBests, critBestsName);
    end
    
    if  abs(crit1)<eps
        UEflows=flow;
        critLog = readmatrix(critLogName);
        critLog(L+1,:) = [crit1,crit2];
        writematrix(critLog, critLogName);
        fprintf('First convergence check met, iter=%d\n',L')
        break
    end
    
    %=============== Step 3: update the flows===================
    % flow=flow+ (1/L)*flow_p;
    % Find step size alpha
    
    fun = @(step)BeckmannMin_UE(step,density,density_y,time,params,criticalDensity);
    
    step_new=fminbnd(fun,0,1);
    %step_new = max(step_new, 2/(L+2));

    % % Improved step size calculation
    % step_numerator = sum((flow_p - flow) .* traveltime);
    % step_denominator = sum((flow_p - flow).^2 .* traveltime);
    % % Add a small perturbation to prevent division by zero
    % step_new = min(1.0, max(0.0, step_numerator / (step_denominator + 1e-10)));
    
    % step_new=1/L;
    flow = flow + step_new * flow_p;
    density = density + step_new * density_p;
    
    %=======Step 4: Stopping Cirterion============
    traveltime = BPR_density_smooth(time,density,criticalDensity,params);
    
    topo_graph_input.Edges.Weight=traveltime;
    
    %T_new=flow'*traveltime;
    T_new=flow'*traveltime;
    
    crit2=abs(T_new-LBD)/LBD;

    % %disp(['Iteration: ', num2str(L), ' Crit1: ', num2str(crit1), ' Crit2: ', num2str(crit2)])
    disp(['Iteration: ', num2str(L), ' Step Size: ', num2str(step_new)])
    disp(['Crit1: ', num2str(crit1), ' Crit2: ', num2str(crit2)])
    disp(['Flow difference norm: ', num2str(norm(flow_p))])
    disp(['Travel Time Update: ', num2str(mean(traveltime))])

    if (crit1 <= crit1Best) && (crit2 <= crit2Best)
        crit1Best = crit1;
        crit2Best = crit2;
        UEflowsBest = flow;
        iter = L;
        LBDBest = LBD;
        critBests = readmatrix(critBestsName);
        critBests(size(critBests,1)+1,:) = [crit1Best,crit2Best,iter];
        writematrix(critBests, critBestsName);
    end
    
    if  abs(crit2)<eps
        Xa_Gi=zeros(length(flow),length(OD_list));
        for i=1:height(topo_graph_input.Nodes)
            [~,~,E] = shortestpathtree(topo_graph_input,i,'OutputForm','cell');
            E_store{i}=E;
        end
    
        for i=1:length(OD_list)
            s=OD_list(i,3);
            t=OD_list(i,4);
            E1=E_store{s};
            edgepath=E1{t};
            Xa_Gi(edgepath,i)=1;
        end
        UEflows=flow;
        critLog = readmatrix(critLogName);
        critLog(L+1,:) = [crit1,crit2];
        writematrix(critLog, critLogName);
        fprintf('Second convergence check met, iter=%d\n',L)
        break
        
    end

    if mod(L,100) == 0
        fid = fopen(txtName,'a+');
        fprintf(string(datetime('now')) + ': completed iteration %d.\n', L);
        fclose(fid);
    end

    critLog = readmatrix(critLogName);
    critLog(L+1,:) = [crit1,crit2];
    writematrix(critLog, critLogName);

end

end
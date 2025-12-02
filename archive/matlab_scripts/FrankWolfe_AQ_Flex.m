function [POflows,crit1,crit2,L,Xa_Gi,critLog,critBests,POflowsBest,crit1Best,crit2Best,iter,LBD,LBDBest]=FrankWolfe_AQ_Flex(demand,topo_graph,OD_list,eps,capacity,criticalDensity,params,stepbreak,txtName,critLogName,critBestsName)

    %Algorithm written according to Patriksson (2015)
    
    %Step 0- Initialisation
    
    topo_graph_input=topo_graph;
    
    flow0=zeros(size(topo_graph.Edges,1),1);
    density0 = flow0 .* criticalDensity./capacity;
    
    traveltime0=NO2_Function(density0, params);
    
    topo_graph_input.Edges.Weight=traveltime0;
    
    % calculate the shortest path for each OD pair to assign all the demand to
    flow=flow0;
    density = density0;
    
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
    
    density = flow .* criticalDensity./capacity;
    
    LBD=0;
    L=0;
    
    flow_y=zeros(length(flow),1);
    density_y = flow_y .* criticalDensity./capacity;

    crit1=inf;
    crit2=inf;

    crit1Best=crit1;
    crit2Best=crit2;
    POflowsBest=flow;
    iter=0;

    %======Step 1 - Solve linear programming problem/ search
    %direction=======
    traveltime=NO2_Function(density, params);
    
    topo_graph_input.Edges.Weight=traveltime*2;
    
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
        
            POflows=flow;
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
        density_y = flow_y .* criticalDensity./capacity;
        density_p = density_y - density;
        %==================== step 2: Convergence Check===================
        
        gradT=NO2_Function(density, params)*2;
        
        T=flow'*(traveltime);
        T_Bar=T+gradT'*flow_p;
        
        LBD=max([LBD,T_Bar]);
        
        crit1=abs(T-LBD)/LBD;

        if (crit1 <= crit1Best) && (crit2 <= crit2Best)
            crit1Best = crit1;
            crit2Best = crit2;
            POflowsBest = flow;
            iter = L;
            LBDBest = LBD;
            critBests = readmatrix(critBestsName);
            critBests(size(critBests,1)+1,:) = [crit1Best,crit2Best,iter];
            writematrix(critBests, critBestsName);
        end
        
        if  abs(crit1)<eps
            POflows=flow;
            critLog = readmatrix(critLogName);
            critLog(L+1,:) = [crit1,crit2];
            writematrix(critLog, critLogName);
            fprintf('First convergence check met, iter=%d\n',L')
            break
        end
        
        %=============== Step 3: update the flows===================
        % flow=flow + (1/L)*flow_p;
        % Find step size
        
        fun = @(step)BeckmannMin_AQ(step,density,density_y,params);
        step_new=fminbnd(fun,0,1);
        %step_new = max(step_new, 2/(L+2));

        %numer = sum(density_p.*(2*m*density));
        %denom = 2*sum(m*density_p.^2);
        %step_new = numer/denom;
        
        % step_new=1/L;
        flow=flow + step_new * flow_p;
        density=density + step_new * density_p;
        
        %=======Step 4: Stopping Cirterion============
        traveltime=NO2_Function(density, params);
        topo_graph_input.Edges.Weight=traveltime*2;
        
        T_new=flow'*traveltime;
       
        crit2=abs(T_new-LBD)/LBD;

        % disp(['Iteration: ', num2str(L), ' Step Size: ', num2str(step_new)])
        % disp(['Crit1: ', num2str(crit1), ' Crit2: ', num2str(crit2)])
        % disp(['Flow difference norm: ', num2str(norm(flow_p))])
        % disp(['Travel Time Update: ', num2str(mean(traveltime))])

        if (crit1 <= crit1Best) && (crit2 <= crit2Best)
            crit1Best = crit1;
            crit2Best = crit2;
            POflowsBest = flow;
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
            POflows=flow;
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
function [UEflows,crit1,crit2,L,Xa_Gi,critLog,critBests,UEflowsBest,crit1Best,crit2Best,iter,LBD,LBDBest] = ...
    FrankWolfe_UE_Flex(demand,time,topo_graph,OD_list,eps,capacity,criticalDensity,params,stepbreak,txtName,critLogName,critBestsName)

topo_graph_input = topo_graph;

num_edges = size(topo_graph.Edges,1);
num_ods   = length(demand);

od_orig = OD_list(:,3);
od_dest = OD_list(:,4);

flow0    = zeros(num_edges,1);
density0 = flow0 .* criticalDensity ./ capacity;

traveltime0 = BPR_density_smooth(time,density0,criticalDensity,params);
topo_graph_input.Edges.Weight = traveltime0;

flow = flow0;

E_store = shortest_paths_for_od_origins(topo_graph_input, od_orig);
for k = 1:num_ods
    edgepath = E_store{od_orig(k)}{od_dest(k)};
    flow(edgepath) = flow(edgepath) + demand(k);
end

LBD   = 0;
L     = 0;
crit1 = inf;
crit2 = inf;

crit1Best   = crit1;
crit2Best   = crit2;
UEflowsBest = flow;
iter        = 0;
LBDBest     = LBD;

Xa_Gi = zeros(num_edges, num_ods);

critLog   = readmatrix(critLogName);
critBests = readmatrix(critBestsName);

density    = flow .* criticalDensity ./ capacity;
traveltime = BPR_density_smooth(time,density,criticalDensity,params);
topo_graph_input.Edges.Weight = traveltime;

while abs(crit1) > eps || abs(crit2) > eps
    L = L + 1;

    if L == stepbreak
        disp(['Max iterations reached...', num2str(L)])

        Xa_Gi  = zeros(num_edges, num_ods);
        E_store = shortest_paths_for_od_origins(topo_graph_input, od_orig);
        for k = 1:num_ods
            edgepath = E_store{od_orig(k)}{od_dest(k)};
            Xa_Gi(edgepath, k) = 1;
        end

        UEflows = flow;
        break
    end

    flow_y = zeros(num_edges,1);

    E_store = shortest_paths_for_od_origins(topo_graph_input, od_orig);
    for k = 1:num_ods
        edgepath = E_store{od_orig(k)}{od_dest(k)};
        flow_y(edgepath) = flow_y(edgepath) + demand(k);
    end

    flow_p   = flow_y - flow;
    density  = flow    .* criticalDensity ./ capacity;
    density_y = flow_y .* criticalDensity ./ capacity;
    density_p = density_y - density;

    gradT = BPR_density_smooth(time,density,criticalDensity,params);
    T     = flow' * traveltime;
    T_Bar = T + gradT' * flow_p;

    LBD   = max([LBD, T_Bar]);
    crit1 = abs(T - LBD) / LBD;

    [crit1Best,crit2Best,UEflowsBest,iter,LBDBest,critBests] = ...
        update_best_solution(crit1,crit2,flow,L,LBD, ...
                             crit1Best,crit2Best,UEflowsBest,iter,LBDBest, ...
                             critBests);

    if (abs(crit1) < eps) && (L > 100)
        UEflows = flow;
        critLog(L+1,:) = [crit1, crit2];
        fprintf('First convergence check met, iter=%d\n', L)
        break
    end

    fun      = @(step) BeckmannMin_UE(step,density,density_y,time,params,criticalDensity);
    step_new = fminbnd(fun,0,1);

    flow    = flow    + step_new * flow_p;
    density = density + step_new * density_p;

    traveltime = BPR_density_smooth(time,density,criticalDensity,params);
    topo_graph_input.Edges.Weight = traveltime;

    T_new = flow' * traveltime;
    crit2 = abs(T_new - LBD) / LBD;

    [crit1Best,crit2Best,UEflowsBest,iter,LBDBest,critBests] = ...
        update_best_solution(crit1,crit2,flow,L,LBD, ...
                             crit1Best,crit2Best,UEflowsBest,iter,LBDBest, ...
                             critBests);

    if (abs(crit2) < eps) && (L > 100)
        Xa_Gi  = zeros(num_edges, num_ods);
        E_store = shortest_paths_for_od_origins(topo_graph_input, od_orig);
        for k = 1:num_ods
            edgepath = E_store{od_orig(k)}{od_dest(k)};
            Xa_Gi(edgepath, k) = 1;
        end

        UEflows = flow;
        critLog(L+1,:) = [crit1, crit2];
        fprintf('Second convergence check met, iter=%d\n', L)
        break
    end

    if mod(L,10) == 0
        disp(['Iteration: ', num2str(L), ' Crit1: ', num2str(crit1), ' Crit2: ', num2str(crit2)])
        fid = fopen(txtName,'a+');
        fprintf(string(datetime('now')) + ': completed iteration %d.\n', L);
        fclose(fid);
    end

    critLog(L+1,:) = [crit1, crit2];
end

writematrix(critLog,   critLogName);
writematrix(critBests, critBestsName);

end

function E_store = shortest_paths_for_od_origins(topo_graph_input, od_orig)
    num_nodes  = height(topo_graph_input.Nodes);
    E_store    = cell(num_nodes,1);
    od_orig_unique = unique(od_orig);

    for j = 1:numel(od_orig_unique)
        s = od_orig_unique(j);
        [~,~,E] = shortestpathtree(topo_graph_input, s, 'OutputForm', 'cell');
        E_store{s} = E;
    end
end

function [crit1Best,crit2Best,UEflowsBest,iter,LBDBest,critBests] = ...
    update_best_solution(crit1,crit2,flow,L,LBD, ...
                         crit1Best,crit2Best,UEflowsBest,iter,LBDBest, ...
                         critBests)

    if (crit1 <= crit1Best) && (crit2 <= crit2Best)
        crit1Best   = crit1;
        crit2Best   = crit2;
        UEflowsBest = flow;
        iter        = L;
        LBDBest     = LBD;
        critBests(end+1,:) = [crit1Best, crit2Best, iter];
    end
end

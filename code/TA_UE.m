clc; clear; close all;

distance_tolerance = 58.6;

data_root_dir   = 'C:/Users/elp24vs/Documents/Samantha Project/no2d/data/';
input_dir_path  = [data_root_dir 'inputs/'];
output_dir_path = [data_root_dir 'outputs/'];

%% LOAD NETWORK DATA
edge_table = readtable([input_dir_path 'edges.csv']);
node_table = readtable([input_dir_path 'nodes.csv']);

from_nodes  = edge_table{:, 1} + 1;
to_nodes    = edge_table{:, 2} + 1;
edge_weight = edge_table{:, 11};

road_network = digraph(from_nodes, to_nodes, edge_weight);

road_network.Nodes.Lon  = node_table.x;
road_network.Nodes.Lat  = node_table.y;
road_network.Nodes.LSOA = node_table.LSOA;

road_network.Edges.Speedlim        = edge_table{:, 17};
road_network.Edges.Lanes           = edge_table{:, 18};
road_network.Edges.Width           = edge_table{:, 19};
road_network.Edges.Capacity        = edge_table{:, 23};
road_network.Edges.CriticalDensity = edge_table{:, 26};
road_network.Edges.AvgSpeed        = edge_table{:, 25};

OD_list = readmatrix([input_dir_path 'OD_list_tol' num2str(distance_tolerance) '.csv']);
OD_list = OD_list(2:end, :);

OD_list(:, 3:4) = OD_list(:, 3:4) + 1;

demand = readmatrix([input_dir_path 'demand.csv']);
demand = demand(2:end);

%% REMOVE INTRA-LSOA DEMAND
is_intra_lsoa = OD_list(:, 1) == OD_list(:, 2);
OD_list(is_intra_lsoa, :) = [];
demand(is_intra_lsoa)    = [];

capacity                = road_network.Edges.Capacity;
critical_density        = road_network.Edges.CriticalDensity;
road_length_km          = road_network.Edges.Weight ./ 1000;
speedlimit_km_per_h     = road_network.Edges.Speedlim;
free_flow_travel_time_h = road_length_km ./ speedlimit_km_per_h;

number_of_graph_edges = numedges(road_network);

% BPR parameters (uniform)
alpha = 0.15 * ones(number_of_graph_edges, 1);
beta  = 4    * ones(number_of_graph_edges, 1);

bpr_params = [alpha, beta, zeros(number_of_graph_edges, 1)]; % last col = epsilon (0 here)

eps       = 1e-5;
steplimit = 200;

suffix = '_test';

txtName = [data_root_dir 'Out_TA_HPC_UE' suffix '.txt'];
s = sprintf('File created: %s.\n', string(datetime('now')));
writematrix(s, txtName);

critLogName   = [data_root_dir 'All_crit1_crit2_UE' suffix '.csv'];
critBestsName = [data_root_dir 'Best_crit1_crit2_UE' suffix '.csv'];

writematrix(inf(steplimit + 1, 2), critLogName);
writematrix([inf, inf, inf], critBestsName);

%% TRAFFIC ASSIGNMENT MODEL
tic

TimeBinPeriods = "DAY";  % placeholder; length can be > 1 in future
number_of_time_bins  = numel(TimeBinPeriods);

UEflows     = zeros(number_of_graph_edges, number_of_time_bins);
UEflowsBest = zeros(number_of_graph_edges, number_of_time_bins);

for i = 1:number_of_time_bins
    disp("Time bin ..." + string(i));
    disp('Starting user-equilibrium Frank-Wolfe...');

    [UEflows(:, i), ...
        crit1_UE, crit2_UE, L_UE, ...
        ~, critLog, critBests, ... %#ok<NASGU>
        UEflowsBest(:, i), ...
        crit1_UE_Best, crit2_UE_Best, ...
        iter_UE, LBD_UE, LBD_UE_Best] = ...
        FrankWolfe_UE_Flex( ...
            demand, ...
            free_flow_travel_time_h, ...
            road_network, ...
            OD_list, ...
            eps, ...
            capacity, ...
            critical_density, ...
            bpr_params, ...
            steplimit, ...
            txtName, ...
            critLogName, ...
            critBestsName);

end

toc

%% WRITE RESULTS
writematrix(UEflows,      [output_dir_path 'UE_flow'       suffix '.csv']);
writematrix(UEflowsBest,  [output_dir_path 'UE_flow_best'  suffix '.csv']);

writematrix([crit1_UE,     crit2_UE],     [output_dir_path 'UE_crit1and2'      suffix '.csv']);
writematrix([crit1_UE_Best, crit2_UE_Best], [output_dir_path 'UE_crit1and2_best' suffix '.csv']);

writematrix(L_UE,    [output_dir_path 'UE_L'      suffix '.csv']);
writematrix(iter_UE, [output_dir_path 'UE_L_best' suffix '.csv']);

writematrix(LBD_UE,     [output_dir_path 'UE_LBD'      suffix '.csv']);
writematrix(LBD_UE_Best,[output_dir_path 'UE_LBD_best' suffix '.csv']);

clc; clear; close all;
tol = 58.6;

%% DEFINE CONSTANTS %%
parentDir = ''; % INSERT MAIN DIRECTORY HERE
inputDir  = [parentDir '../data/inputs/'];
outputDir = [parentDir '../data/outputs/'];

%% LOAD DATA (TABLE-FIRST, KEEP EDGE ATTRIBUTES ALIGNED) %%
E = readtable(fullfile(inputDir, "edges.csv"));
N = readtable(fullfile(inputDir, "nodes.csv"));

highway = E.highway; % keep as table column (string/cell)

s = E.u + 1;
t = E.v + 1;

edgeTable = table( ...
    [s t], ...
    E.length, ...
    E.speedlim, ...
    E.lanes, ...
    E.width, ...
    E.capacity, ...
    E.criticalDensity, ...
    'VariableNames', {'EndNodes','Weight','Speedlim','Lanes','Width','Capacity','CriticalDensity'} ...
);

G = digraph(edgeTable);

G.Nodes.Lon  = N.x;
G.Nodes.Lat  = N.y;
G.Nodes.LSOA = N.LSOA;

%% LOAD OD + DEMAND %%
OD_list = readmatrix([inputDir ['OD_list_tol' num2str(tol) '.csv']]);
OD_list = OD_list(2:end,:);

% PLUS ONE DUE TO PYTHON INDEXING IN ORIGINAL FILE
OD_list(:,3) = OD_list(:,3) + 1;
OD_list(:,4) = OD_list(:,4) + 1;

demand = readmatrix([inputDir 'demand.csv']);
demand = demand(2:end);

%% REMOVE INTRA-LSOA DEMAND %%
inds = find(OD_list(:,1) == OD_list(:,2));
OD_list(inds,:) = [];
demand(inds) = [];

%% ASSIGN BPR FUNCTION PARAMETERS PER EDGE & OTHER MEASURES %%
capacity           = G.Edges.Capacity;
criticalDensity    = G.Edges.CriticalDensity;
distance_km        = G.Edges.Weight ./ 1000;
speedlimit_kmh     = G.Edges.Speedlim;
free_flow_travel_h = distance_km ./ speedlimit_kmh;

% **Literature-Based Approach**
% alpha range: 0-1, generally 0.15
% beta range: 2-5, generally 4
% - Urban roads: alpha = 0.15, beta = 4.0
% - Highway: alpha = 0.75, beta = 4.0
% - Rural roads: alpha = 0.5, beta = 4.0

nEdges = height(G.Edges);

roadClass = cell(nEdges, 1);
for ii = 1:nEdges
    h = string(highway(ii));
    if contains(h,'motorway') || contains(h,'trunk')
        roadClass(ii) = {'highway'};
    else
        if contains(h,'tertiary') || contains(h,'unclassified')
            roadClass(ii) = {'rural'};
        else
            roadClass(ii) = {'urban'};
        end
    end
end

% for the BPR function
alpha = repelem(0.15, nEdges)';
beta  = repelem(4,    nEdges)';

params = [alpha, beta, repmat(0, [nEdges, 1])]; % final value is convergence epsilon

TimeBinPeriods = "DAY";

% FW Options
eps = 1e-5; % PATIL ET AL. recommends 10^-5
steplimit = 125000;

UEflows = [];
UEflowsBest = [];

%% CREATE ITERATIVELY EDITED TXT & CSV OUTPUT FILES %%
txtName = [parentDir 'Out_TA_HPC_UE.txt'];
s = sprintf('File created: ' + string(datetime('now')) + '.\n');
writematrix(s, txtName);

critLogName = [parentDir 'All_crit1_crit2_UE.csv'];
writematrix(inf(steplimit+1,2), critLogName);

critBestsName = [parentDir 'Best_crit1_crit2_UE.csv'];
writematrix([inf,inf,inf], critBestsName);

%% BEGIN TRAFFIC ASSIGNMENT MODEL %%
tic
for i = 1:length(TimeBinPeriods)
    disp(strcat('Time bin ...', string(i)))
    disp('Starting user-equilibrium Frank-Wolfe...')

    [UEflows(:,i),crit1_UE,crit2_UE,L_UE,~,critLog,critBests, ...
     UEflowsBest(:,i),crit1_UE_Best,crit2_UE_Best,iter_UE,LBD_UE,LBD_UE_Best] = ...
        FrankWolfe_UE_Flex( ...
            demand, free_flow_travel_h, G, OD_list, eps, ...
            capacity, criticalDensity, params, steplimit, ...
            txtName, critLogName, critBestsName ...
        );
end
toc

%% WRITE RESULTS %%
writematrix(UEflows,     [outputDir 'UE_flow.csv']);
writematrix(UEflowsBest, [outputDir 'UE_flow_best.csv']);

writematrix([crit1_UE, crit2_UE],             [outputDir 'UE_crit1and2.csv']);
writematrix([crit1_UE_Best, crit2_UE_Best],   [outputDir 'UE_crit1and2_best.csv']);

writematrix(L_UE,    [outputDir 'UE_L.csv']);
writematrix(iter_UE, [outputDir 'UE_L_best.csv']);

writematrix(LBD_UE,      [outputDir 'UE_LBD.csv']);
writematrix(LBD_UE_Best, [outputDir 'UE_LBD_best.csv']);

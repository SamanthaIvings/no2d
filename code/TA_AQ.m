clc; clear; close all;
tol = 58.6;

%% DEFINE CONSTANTS %%
parentDir = ''; % INSERT MAIN DIRECTORY HERE
inputDir = [parentDir 'data/inputs/'];
outputDir = [parentDir 'data/outputs/'];

%% LOAD DATA %%
edges = readmatrix([inputDir 'edges.csv']);
nodes = readtable([inputDir 'nodes.csv']);

G = digraph(edges(:,1)+1, edges(:,2)+1, edges(:,11));
G.Nodes.Lon = nodes.x; G.Nodes.Lat = nodes.y;
G.Nodes.LSOA = nodes.LSOA;
G.Edges.Speedlim = edges(:,17);
G.Edges.Lanes = edges(:,18);
G.Edges.Width = edges(:,19);
G.Edges.Capacity = edges(:,23);
G.Edges.CriticalDensity = edges(:,26);
G.Edges.AvgSpeed = edges(:,25);

lDists = readtable([inputDir ['lDists_tol' num2str(tol) '.csv']]);

OD_list = readmatrix([inputDir ['OD_list_tol' num2str(tol) '.csv']]);
OD_list = OD_list(2:end,:);
% PLUS ONE DUE TO PYTHON INDEXING IN ORIGINAL FILE
OD_list(:,3) = OD_list(:,3) + 1;
OD_list(:,4) = OD_list(:,4) + 1;

OD_matrix = readmatrix([inputDir 'OD_matrix.csv']);
OD_matrix = OD_matrix(2:end,:);

OD_names = readcell([inputDir 'OD_names.csv']);
OD_names(1) = [];

demand = readmatrix([inputDir 'demand.csv']);
demand = demand(2:end);

n = length(OD_names);

numHours = 2+3; % how many hours of data is in this OD timeband

%% REMOVE INTRA-LSOA DEMAND %%
inds = find(OD_list(:,1)==OD_list(:,2));

OD_list(inds,:) = [];
demand(inds) = [];

%% ASSIGN BPR FUNCTION PARAMETERS PER EDGE & OTHER MEASURES %%
carsAndTaxis2019 = 0.78;
carTaxiProportion = 0.95;
demandTot=demand/(carsAndTaxis2019*carTaxiProportion);

% a,b
%params = [3.42, 0.0079];
% m
params = 0.01369839;

capacity = G.Edges.Capacity;
criticalDensity = G.Edges.CriticalDensity;

% road length in meters to km
distance = G.Edges.Weight ./ 1000;
% speedlim in km/hour
speed = G.Edges.Speedlim;
% free-flow travel times in km/hour
time = distance./speed;

SuperNode = G.Nodes; % table of the nodes
SuperEdge = G.Edges; % table of the edges

elapsedTime=[];
res_choice_count=1;
TimeBinPeriods="DAY";%numHours;

% FW Options
eps=1e-5; %  PATIL ET AL. recommends 10^-5
steplimit=125000;
plotter=0;
%
UEflows=[];
POflows=[];

CFNE=[];
CFSO=[];


CFNE_edge=[];
CFSO_edge=[];
user_tt=[];
system_tt=[];

%% CREATE ITERATIVELY EDITED TXT & CSV OUTPUT FILES %%
txtName = [parentDir 'Out_TA_HPC_PO.txt'];
s = sprintf('File created: ' + string(datetime('now')) + '.\n');
writematrix(s, txtName);

critLogName = [parentDir 'All_crit1_crit2_PO.csv'];
writematrix(inf(steplimit+1,2), critLogName);

critBestsName = [parentDir 'Best_crit1_crit2_PO.csv'];
writematrix([inf,inf,inf], critBestsName);

%% BEGIN TRAFFIC ASSIGNMENT MODEL %%
% topo_graph = G;
% stepbreak = steplimit;
tic
for i=1:length(TimeBinPeriods)
    disp(strcat('Time bin ...',string(i)))

    disp('Starting pollution-optimal Frank-Wolfe...')
    [POflows(:,i),crit1_PO,crit2_PO,L_PO,~,critLog,critBests,POflowsBest(:,i),crit1_PO_Best,crit2_PO_Best,iter_PO,LBD_PO,LBD_PO_Best]=FrankWolfe_AQ_Flex(demand,G,OD_list,eps,capacity,criticalDensity,params,steplimit,txtName,critLogName,critBestsName);
end
toc

%% WRITE RESULTS %%
% flows
writematrix(POflows, [outputDir 'PO_flow.csv']);
writematrix(POflowsBest, [outputDir 'PO_flow_best.csv']);

% crit values
writematrix([crit1_PO, crit2_PO], [outputDir 'PO_crit1and2.csv']);
writematrix([crit1_PO_Best, crit2_PO_Best], [outputDir 'PO_crit1and2_best.csv']);

% L (iteration at which the code reached optimum and/or timed out)
writematrix(L_PO, [outputDir 'PO_L.csv']);
writematrix(iter_PO, [outputDir 'PO_L_best.csv']);

% LBD
writematrix(LBD_PO, [outputDir 'PO_LBD.csv']);
writematrix(LBD_PO_Best, [outputDir 'PO_LBD_best.csv']);


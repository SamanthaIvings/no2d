clc; clear; close all;
tol = 58.6;

%% DEFINE CONSTANTS %%
parentDir = ''; % INSERT MAIN DIRECTORY HERE
inputDir = [parentDir 'inputs/'];
outputDir = [parentDir 'outputs/'];

%% LOAD DATA %%
edges = readtable([inputDir 'edges.csv']);
nodes = readtable([inputDir 'nodes.csv']);

highway = edges.highway;
edges = readmatrix([inputDir 'edges.csv']);

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
capacity = G.Edges.Capacity;
criticalDensity = G.Edges.CriticalDensity;
avgSpeed = G.Edges.AvgSpeed;

% road length in meters to km
distance = G.Edges.Weight ./ 1000;
% speedlim in km/hour
speed = G.Edges.Speedlim;
% free-flow travel times in km/hour
time = distance./speed;

% **Literature-Based Approach**
% alpha range: 0-1, generally 0.15
% beta range: 2-5, generally 4
% - Urban roads: alpha = 0.15, beta = 4.0
% - Highway: alpha = 0.75, beta = 4.0
% - Rural roads: alpha = 0.5, beta = 4.0

roadClass = cell(height(edges),1);
for ii = 1:height(edges)
    if contains(highway(ii),'motorway') | contains(highway(ii),'trunk')
        roadClass(ii) = {'highway'};
    else
        if contains(highway(ii),'tertiary') | contains(highway(ii),'unclassified')
            roadClass(ii) = {'rural'};
        else
            roadClass(ii) = {'urban'};
        end
    end
end

% for the BPR function
alpha = repelem(0.15,height(G.Edges.EndNodes))';
beta = repelem(4,height(G.Edges.EndNodes))';
% alpha = zeros(height(edges),1);
% beta = zeros(height(edges),1);
% for ii = 1:height(edges)
%     if contains(roadClass{ii},'highway')
%         alpha(ii) = 0.75; beta(ii) = 4;
%     else
%         if contains(roadClass{ii},'rural')
%             alpha(ii) = 0.5; beta(ii) = 4;
%         else
%             alpha(ii) = 0.15; beta(ii) = 4;
%         end
%     end
% end
params = [alpha, beta, repmat(0, [height(edges),1])]; % final value is convergence epsilon

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
txtName = [parentDir 'Out_TA_HPC_UE.txt'];
s = sprintf('File created: ' + string(datetime('now')) + '.\n');
writematrix(s, txtName);

critLogName = [parentDir 'All_crit1_crit2_UE.csv'];
writematrix(inf(steplimit+1,2), critLogName);

critBestsName = [parentDir 'Best_crit1_crit2_UE.csv'];
writematrix([inf,inf,inf], critBestsName);

%% BEGIN TRAFFIC ASSIGNMENT MODEL %%
tic
% topo_graph = G;
% stepbreak = steplimit;
for i=1:length(TimeBinPeriods)
    disp(strcat('Time bin ...',string(i)))

    disp('Starting user-equilibrium Frank-Wolfe...')
    [UEflows(:,i),crit1_UE,crit2_UE,L_UE,~,critLog,critBests,UEflowsBest(:,i),crit1_UE_Best,crit2_UE_Best,iter_UE,LBD_UE,LBD_UE_Best]=FrankWolfe_UE_Flex(demand,time,G,OD_list,eps,capacity,criticalDensity,params,steplimit,txtName,critLogName,critBestsName);
end
toc

%% WRITE RESULTS %%
% flows
writematrix(UEflows, [outputDir 'UE_flow.csv']);
writematrix(UEflows, [outputDir 'UE_flow_best.csv']);

% crit values
writematrix([crit1_UE, crit2_UE], [outputDir 'UE_crit1and2.csv']);
writematrix([crit1_UE_Best, crit2_UE_Best], [outputDir 'UE_crit1and2_best.csv']);

% L (iteration at which the code reached optimum and/or timed out)
writematrix(L_UE, [outputDir 'UE_L.csv']);
writematrix(iter_UE, [outputDir 'UE_L_best.csv']);

% LBD
writematrix(LBD_UE, [outputDir 'UE_LBD.csv']);
writematrix(LBD_UE_Best, [outputDir 'UE_LBD_best.csv']);

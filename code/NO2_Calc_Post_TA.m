clc; clear; close all;
tol = 58.6;

%% DEFINE DIRECTORIES %%
parentDir = ''; % INSERT WORKING DIRECTORY
inputDir = [parentDir 'inputs\'];
outputDir = [parentDir 'outputs\'];

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

% demand = readmatrix([inputDir 'demand.csv']);
% demand = demand(2:end);

UE_flow = readmatrix([outputDir 'UE_flow_best.csv']);
PO_flow = readmatrix([outputDir 'PO_flow_best.csv']);

%% CREATE GRAPH %%
G = digraph(edges(:,1)+1, edges(:,2)+1, edges(:,11));
G.Nodes.Lon = nodes.x; G.Nodes.Lat = nodes.y;
G.Nodes.LSOA = nodes.LSOA;
G.Edges.Speedlim = edges(:,17);
G.Edges.Lanes = edges(:,18);
G.Edges.Width = edges(:,19);
G.Edges.Capacity = edges(:,23);
G.Edges.CriticalDensity = edges(:,26);
G.Edges.AvgSpeed = edges(:,25);

%% DEFINE PARAMETERS %%
% m,thresh
params = [0.0136983940720969, 240.125];

capacity = G.Edges.Capacity;
criticalDensity = G.Edges.CriticalDensity;
avgSpeed = G.Edges.AvgSpeed;

alpha = 0.15;%BPR_params(:,1);%0.15;
beta = 4;%BPR_params(:,2);%4;

R = 8.31;
MM = 46.01;
T = 13.497 + 273.15; % average temperature
P = 1003.703 * 100; % average pressure
convFactor = (MM*P)/(R*T*1000);

% minimum NO2 partial dependency value for zero-offsetting
minRes = 15.23094;

% road length in meters to km
distance = G.Edges.Weight ./ 1000;
% speed in km/hour
speed = G.Edges.Speedlim;
% free-flow travel times in km/hour
freeFlowTT = distance ./ speed;

n = height(G.Edges);

fprintf('Total flows: UE=%f, PO=%f\n',sum(UE_flow),sum(PO_flow))
%fprintf('Total flows: UE=%f, PO=%f, PO old=%f\n',sum(UE_flow),sum(PO_flow),sum(PO_flow2))

%% HISTOGRAMS %%
% figure(); histogram(capacity); title('Capacity');
% figure(); histogram(criticalDensity); title('Critical density');
% figure(); histogram(freeFlowTT*60); title('Travel time (mins/km)');

%% TRAVEL TIMES %%
UE_TT = freeFlowTT .* (1 + alpha .* (UE_flow./capacity).^beta);
PO_TT = freeFlowTT .* (1 + alpha .* (PO_flow./capacity).^beta);

TT_diff = PO_TT - UE_TT;

fprintf('Mean travel times: UE=%f, PO=%f\n', mean(UE_TT)*60,mean(PO_TT)*60)

%% CONVERT FLOW TO DENSITY %%
UE_density = UE_flow .* criticalDensity./capacity;
PO_density = PO_flow .* criticalDensity./capacity;

% UE_density = UE_flow ./ avgSpeed;
% PO_density = PO_flow ./ avgSpeed;

UE_density_below_thresh = length(find(UE_density <= params(2)));
UE_density_below_thresh_percent = UE_density_below_thresh*100/length(UE_flow);

PO_density_below_thresh = length(find(PO_density <= params(2)));
PO_density_below_thresh_percent = PO_density_below_thresh*100/length(PO_flow);

fprintf('Total densities: UE=%f, PO=%f\n',sum(UE_density),sum(PO_density))

%% ASSOCIATE NO2 %%
UE_NO2 = params(1) .* UE_density;
PO_NO2 = params(1) .* PO_density;

fprintf('Total NO2: UE=%f, PO=%f\n',sum(UE_NO2),sum(PO_NO2))
%fprintf('Total NO2: UE=%f, PO=%f, PO old=%f\n',sum(UE_NO2),sum(PO_NO2),sum(PO_NO22))

%% PRICE OF ANARCHY %%
% user equilibrium calculation of the cost functions
CFNE = UE_flow' * NO2_Function(UE_density,params)*convFactor;
% pollution optimal calculation of the cost functions
CFSO = PO_flow' * NO2_Function(PO_density,params)*convFactor;

POA = CFNE/CFSO

% user equilibrium calculation of the cost functions
CFNE_TT = UE_flow' * BPR_density(freeFlowTT,UE_density,criticalDensity,[repelem(alpha,length(UE_density)),repelem(beta,length(UE_density))]);
% pollution optimal calculation of the cost functions
CFSO_TT = PO_flow' * BPR_density(freeFlowTT,PO_density,criticalDensity,[repelem(alpha,length(UE_density)),repelem(beta,length(UE_density))]);

POA_TT = CFNE_TT/CFSO_TT

sum(UE_TT)./sum(PO_TT)

%% FLOW/CAPACITY RATIOS %%
UE_rat = UE_flow ./ capacity;
PO_rat = PO_flow ./ capacity;

UE_exceeds_cap = length(find(UE_rat>1));
PO_exceeds_cap = length(find(PO_rat>1));
UE_exceeds_cap_percent = UE_exceeds_cap/length(UE_flow)*100;
PO_exceeds_cap_percent = PO_exceeds_cap/length(PO_flow)*100;

mean(UE_rat)
mean(PO_rat)

%% NO2 IN UG/M3 %%
UE_NO2_ugm3 = UE_NO2 .* convFactor;
PO_NO2_ugm3 = PO_NO2 .* convFactor;

min(UE_NO2_ugm3)
min(PO_NO2_ugm3)

max(UE_NO2_ugm3)
max(PO_NO2_ugm3)

mean(UE_NO2_ugm3)
mean(PO_NO2_ugm3)

100 - sum(PO_NO2_ugm3)*100./sum(UE_NO2_ugm3)

%%
length(find(UE_NO2_ugm3 * 9.034974 >= 10))
length(find(PO_NO2_ugm3 * 9.034974 >= 10))

length(find(UE_NO2_ugm3 * 9.034974 >= 20))
length(find(PO_NO2_ugm3 * 9.034974 >= 20))

length(find(UE_NO2_ugm3 * 9.034974 >= 10))*100/length(UE_NO2_ugm3)
length(find(PO_NO2_ugm3 * 9.034974 >= 10))*100/length(PO_NO2_ugm3)

length(find(UE_NO2_ugm3 * 9.034974 >= 20))*100/length(UE_NO2_ugm3)
length(find(PO_NO2_ugm3 * 9.034974 >= 20))*100/length(PO_NO2_ugm3)

%% PROPORTION OF NO2 COMPARED TO LIMIT %%
UE_NO2_Tot = UE_NO2_ugm3/0.68;
PO_NO2_Tot = PO_NO2_ugm3/0.68;

UE_NO2_Prop = UE_NO2_Tot/200;
PO_NO2_Prop = PO_NO2_Tot/200;

min(UE_NO2_Prop)
min(PO_NO2_Prop)

max(UE_NO2_Prop)
max(PO_NO2_Prop)

mean(UE_NO2_Prop)
mean(PO_NO2_Prop)

%% WRITE RESULTS %%
writematrix(UE_flow, [outputDir 'UE_flow_out.csv']);
writematrix(PO_flow, [outputDir 'PO_flow_out.csv']);

writematrix(UE_density, [outputDir 'UE_density_out.csv']);
writematrix(PO_density, [outputDir 'PO_density_out.csv']);

writematrix(UE_NO2, [outputDir 'UE_NO2_out.csv']);
writematrix(PO_NO2, [outputDir 'PO_NO2_out.csv']);

writematrix(UE_NO2_Tot, [outputDir 'UE_NO2_Tot_out.csv']);
writematrix(PO_NO2_Tot, [outputDir 'PO_NO2_Tot_out.csv']);

writematrix(UE_NO2_Prop, [outputDir 'UE_NO2_Prop_out.csv']);
writematrix(PO_NO2_Prop, [outputDir 'PO_NO2_Prop_out.csv']);

writematrix(UE_NO2_ugm3, [outputDir 'UE_NO2_ugm3_out.csv']);
writematrix(PO_NO2_ugm3, [outputDir 'PO_NO2_ugm3_out.csv']);

writematrix(UE_TT, [outputDir 'UE_TT_out.csv']);
writematrix(PO_TT, [outputDir 'PO_TT_out.csv']);

%% SUMS %%
fprintf('Final total flows: UE=%f, PO=%f\n',sum(UE_flow),sum(PO_flow))
fprintf('Final total densities: UE=%f, PO=%f\n',sum(UE_density),sum(PO_density))
fprintf('Final total NO2: UE=%f, PO=%f\n',sum(UE_NO2_ugm3),sum(PO_NO2_ugm3))
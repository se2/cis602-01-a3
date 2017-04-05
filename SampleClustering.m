clc
clear
close all
 
% load dataset
load('USPS.mat');
% normalize row feature
fea = NormalizeFea(fea, 1);
 
% the ground truth number of clusters is 10
numberOfCluster = 10;
 
% run Kmeans clustering
% MaxIter is the number iterations of Kmeans, and Replicates is the number
% of repeat times of Kmeans with different initialization
 
 
tic;
[predictLabel, center] = litekmeans(fea, numberOfCluster, 'MaxIter', 100, 'Replicates', 2);
kmeansRunTime = toc;
 
% compute the clustering accuracy
clusteringAcc = accuracy(gnd, predictLabel);
% compute the clustering NMI
clusteringNMI = nmi(gnd, predictLabel);
 
fprintf('the clustering accuracy of Kmeans is %f.\n', clusteringAcc/100);
fprintf('the running time of Kmeans is %f seconds.\n', kmeansRunTime);


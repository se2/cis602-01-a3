clear
clc
% load data
load('PIE.mat')
fea = Data;
gnd = Label;
% load('USPS.mat');
% normalize data
fea = NormalizeFea(fea, 1);

% reduce dim with PCA
options=[];
options.ReducedDim = 500;
[eigvector, eigvalue] = PCA(fea,options);
fea = fea * eigvector;

[nFea, n] = size(fea);

% number of clusters
numCluster = 67;
% numCluster = 10;
% affinity matrix
affinity_matrix =[];
degree_maxtrix = [];

sigma = 1;
% start timing
tic;

for i=1:nFea    
    for j=1:nFea
       dist = norm((fea(i,:) - fea(j,:)));
%        distance is 0 means self-node
        if dist == 0
            affinity_matrix(i,j) = 0;
        else affinity_matrix(i,j) = exp(-dist/(2*sigma^2));
        end          
    end
%     compute degree matrix
    degree_maxtrix(i,i) = sum(affinity_matrix(i,:));
end

% compute Laplacian matrix and normalize
laplacian_maxtrix = NormalizeFea(degree_maxtrix - affinity_matrix);
% laplacian_maxtrix = NormalizeFea(degree_maxtrix.^-0.5 * affinity_matrix * degree_maxtrix.^-0.5);
% compute eigen vectors of Laplacian matrix
[lap_eigVector, lap_eigValue] = eig(laplacian_maxtrix);


% USPS data
% k = 10, acc = ~49~51
% k = 50, acc = 45~53
% k = 100, acc = 42~49
% k = 200, acc = 38~46
% k = 500, acc = 39~45
% k = 1000, acc = 35~43
% k = 1100, acc = 33~37

% PIE data
% k = 50, acc = 36~39
% k = 100, acc = 39~43
% k = 150, acc = 43~46
% k = 200, acc = 41~44
% k = 300, acc = 33~41
% k = 500, acc = 28~31
% first k eigVectos
k = 67;
% filter first k vectors
lap_eigVector = lap_eigVector(:,1:k);
% normalize
lap_eigVector = NormalizeFea(lap_eigVector);
% predict use K-means
[predictLabel] = litekmeans(lap_eigVector, numCluster, 'Replicates', 2);

runningTime = toc;

tic;
[kpredictLabel, center] = litekmeans(fea, 10, 'Replicates', 2);
kmeansRunTime = toc;


% compute the clustering accuracy
clusteringAcc = accuracy(gnd, predictLabel);
kclusteringAcc = accuracy(gnd, kpredictLabel);
% compute the clustering NMI
clusteringNMI = nmi(gnd, predictLabel);
kclusteringNMI = nmi(gnd, kpredictLabel);
% print results
fprintf('the clustering accuracy is %f.\n', clusteringAcc/100);
fprintf('the clustering accuracy of SVM(NMI) is %f.\n', clusteringNMI);
fprintf('the running time is %f seconds.\n', runningTime);

fprintf('the clustering of Kmeans accuracy is %f.\n', kclusteringAcc/100);
fprintf('the clustering accuracy of SVM(NMI) in Kmeans is %f.\n', kclusteringNMI);
fprintf('the running time of Kmeans is %f seconds.\n', kmeansRunTime);
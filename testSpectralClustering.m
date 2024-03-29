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
sub_affinity_matrix = [];
affinity_matrix = zeros(nFea, nFea);
degree_maxtrix = [];
knear = 3;
sigma = 0.5;
% start timing
tic;

for i=1:nFea    
    for j=1:nFea
       dist = norm((fea(i,:)' - fea(j,:)'));
%        distance is 0 means self-node
        if dist == 0
            sub_affinity_matrix(1,j) = 0;
        else sub_affinity_matrix(1,j) = exp(-(dist^2 /(2*sigma^2)));
        end          
    end
% %     choose k-nearest neighbours
    [newFea, index] = sort(sub_affinity_matrix, 'descend');
    newFea = newFea(1,1:knear);
    index = index(1,1:knear);
    affinity_matrix(i,index) = newFea;
%     compute degree matrix
    degree_maxtrix(i) = sum(affinity_matrix(i,:));
end
affinityTime = toc;
fprintf('the running time of computing Affinity and Degree matrix is %f seconds.\n', affinityTime);

tic;
% compute Laplacian matrix and normalize
D = degree_maxtrix.^-.5;
D = diag(D);
laplacian_maxtrix = D * affinity_matrix * D;

% compute first k-eigvectors of Laplacian matrix
k = 67;
[lap_eigVector, lap_eigValue] = eigs(laplacian_maxtrix, k);
eigTime = toc;

fprintf('the running time of computing first k-eigvectors is %fT seconds.\n', eigTime);

tic;
% flip to have eigvector indicates with smallest eigvalue on left
lap_eigVector = fliplr(lap_eigVector);
% normalize row k-eigvectors
lap_eigVector = NormalizeFea(lap_eigVector,1);
% predict use K-means
[predictLabel] = litekmeans(lap_eigVector, numCluster, 'Replicates', 2);

runningTime = toc;

tic;
[kpredictLabel] = litekmeans(fea, numCluster, 'Replicates', 2);
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
fprintf('the running time of kmeans is %f seconds.\n\n', runningTime);

fprintf('the clustering of Kmeans accuracy is %f.\n', kclusteringAcc/100);
fprintf('the clustering accuracy of SVM(NMI) in Kmeans is %f.\n', kclusteringNMI);
fprintf('the running time of Kmeans is %f seconds.\n', kmeansRunTime);
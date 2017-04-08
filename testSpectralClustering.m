clear
clc
% load data
load('PIE.mat')
fea = Data;
gnd = Label;
% normalize data
fea = NormalizeFea(fea, 1);

% reduce dim with PCA
options=[];
options.ReducedDim = 500;
[eigvector, eigvalue] = PCA(fea,options);
fea = fea * eigvector;

[nFea, n] = size(fea);

% affinity matrix
affinity_matrix =[];
degree_maxtrix = [];

sigma = 1;

for i=1:nFea    
    for j=1:nFea
        if i==j
%             no self-node connection
            affinity_matrix(i,j) = 0;
        else 
            dist = norm((fea(i,:) - fea(j,:)),1);
%             distance is 0 means self-node
            if dist == 0
                affinity_matrix(i,j) = 0;
            else affinity_matrix(i,j) = exp(-dist/(2*sigma^2));
            end
            
        end
    end
%     compute degree matrix
    degree_maxtrix(i,i) = sum(affinity_matrix(i,:));
end

% compute Laplacian matrix and normalize
laplacian_maxtrix = NormalizeFea(degree_maxtrix - affinity_matrix);
% compute eigen vectors of Laplacian matrix
[lap_eigVector, lap_eigValue] = eig(laplacian_maxtrix);


% first k eigVectos
k = 800;
% filter first k vectors
lap_eigVector = lap_eigVector(:,1:k);
% normalize
lap_eigVector = NormalizeFea(lap_eigVector);
% predict use K-means with k = 67
[predictLabel, center] = litekmeans(lap_eigVector, 67, 'MaxIter', 100, 'Replicates', 2);

% compute the clustering accuracy
clusteringAcc = accuracy(gnd, predictLabel);
% compute the clustering NMI
clusteringNMI = nmi(gnd, predictLabel);

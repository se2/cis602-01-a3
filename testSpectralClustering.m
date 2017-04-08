clear
clc
% load data
load('USPS.mat')

% normalize data
fea = NormalizeFea(fea, 1);

[nFea, n] = size(fea);

% affinity matrix
affinity_matrix =[];

sigma = 1;

for i=1:nFea    
    for j=1:nFea
%         dist = sqrt((fea(i,1) - fea(j,1))^2 + (fea(i,2) - fea(j,2))^2); 
%         affinity_matrix(i,j) = exp(-dist/(2*sigma^2));
        affinity_matrix(i,j) = 1;
    end
end
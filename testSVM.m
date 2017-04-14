clear
clc
close all

% load dataset
% USPS
% load('USPS.mat');
% PIE
load('PIE.mat');
fea = Data;
gnd = Label;
clear Data;
clear Label;
% normalize row feature
fea = NormalizeFea(fea, 1);

% number of samples in each digit
% USPS data
% numSampleInEachDigit = 1100;

% number of training data
numTrainingInEachDigit = 5;

% get training data and training label
trainIndex = [];
testIndex = [];

% % USPS data
% for i = 0:9
%     trainIndex = [trainIndex, i*numSampleInEachDigit + 1: i* numSampleInEachDigit + numTrainingInEachDigit];
%     testIndex = [testIndex, i*numSampleInEachDigit + numTrainingInEachDigit + 1: (1+i)*numSampleInEachDigit];
% end

% PIE data
for i = 1: n_per
        trainIndex = [trainIndex, (i-1)*n_sub+1: (i-1)*n_sub+numTrainingInEachDigit];
        testIndex = [testIndex, (i-1)*n_sub+numTrainingInEachDigit+1: i*n_sub];
end

%generate training and testing data
trainFea = fea(trainIndex,:);
trainLabel = gnd(trainIndex,:);
testFea = fea(testIndex,:);
testLabel = gnd(testIndex,:);

[nFea, n] = size(trainFea);

%% Linear kernel
% compute model cross-validation with 5-fold
% linear with slack C (deafault 1, 5, 10, 15)
C = [1,5,10,15]
cross_validation = ' -v 5 '
train_options = '-t 0 -c '
model = [];
for i=C
    sub_train_options = [train_options, num2str(i), cross_validation];
    model(i,:) = svmtrain(trainLabel, trainFea, sub_train_options);
    clear sub_train_options
end

% get max model with highest accuracy
[maxValue, maxIndex] = max(model);
train_options = [train_options, num2str(maxIndex)];

% start running time
tic;
% construct model 
linearModel = svmtrain(trainLabel, trainFea, train_options);
% predict using svm
[predictLabel] = svmpredict(testLabel, testFea, linearModel);

% stop running time
runningTime = toc;

% compute accuracy
% clusteringAcc = accuracy(testLabel, predictLabel);
% compute the clustering NMI
clusteringNMI = nmi(testLabel, predictLabel);

%% Polynomial kernel
% % polynomial cross validation 5-fold
% cross_validation = ' -v 5'
% train_options = '-t 1'
% gama = [1/nFea, 0.1, 0.001, 1, 10];
% degree = [2, 3, 4, 5];
% model = [];
% for i=gama
%     sub_gama = [' -g ', num2str(i)];
%     subgama_train_options = [train_options, cross_validation, sub_gama];
%     for j=degree
%         sub_degree = [' -d ', num2str(j)];
%         subdegree_train_options = [subgama_train_options, sub_degree];
%         sub_model = svmtrain(trainLabel, trainFea, subdegree_train_options);
%         model = [model, vertcat(sub_model, i, j)];
%         clear sub_model
%         clear subdegree_train_options
%         clear sub_degree
%     end
%     clear subgama_train_options
%     clear sub_gama
% end
% [maxValue, maxIndex] = max(model(1,:));
% train_options = [train_options, ' -g ', num2str(model(2,maxIndex)), ' -d ', num2str(model(3,maxIndex))];
% 
% % start running time
% tic;
% % construct model 
% polynomialModel = svmtrain(trainLabel, trainFea, train_options);
% 
% % predict using svm
% [predictLabel] = svmpredict(testLabel, testFea, polynomialModel);

% % stop running time
% runningTime = toc;
% 
% % compute accuracy
% % clusteringAcc = accuracy(testLabel, predictLabel);
% % compute the clustering NMI
% clusteringNMI = nmi(testLabel, predictLabel);

%% Gaussion kernel
% % Gaussion cross validation 5-fold
% cross_validation = ' -v 5'
% train_options = '-t 2'
% gama = [1/nFea, 0.1, 0.001, 1, 10];
% C = [1,5,10,15]
% model = [];
% for i=gama
%     sub_gama = [' -g ', num2str(i)];
%     subgama_train_options = [train_options, cross_validation, sub_gama];
%     for j=C
%         sub_C = [' -c ', num2str(j)];
%         subdegree_train_options = [subgama_train_options, sub_C];
%         sub_model = svmtrain(trainLabel, trainFea, subdegree_train_options);
%         model = [model, vertcat(sub_model, i, j)];
%         clear sub_model
%         clear subdegree_train_options
%         clear sub_C
%     end
%     clear subgama_train_options
%     clear sub_gama
% end
% [maxValue, maxIndex] = max(model(1,:));
% train_options = [train_options, ' -g ', num2str(model(2,maxIndex)), ' -c ', num2str(model(3,maxIndex))];
% 
% % start running time
% tic;
% % construct model 
% gaussionModel = svmtrain(trainLabel, trainFea, train_options);
% 
% % predict using svm
% [predictLabel] = svmpredict(testLabel, testFea, gaussionModel);
% 
% % stop running time
% runningTime = toc;
% 
% % compute accuracy
% % clusteringAcc = accuracy(testLabel, predictLabel);
% % compute the clustering NMI
% clusteringNMI = nmi(testLabel, predictLabel);

%% Print result
% 
% fprintf('the clustering accuracy of SVM is %f.\n', clusteringAcc);
% fprintf('the clustering accuracy of SVM(NMI) is %f.\n', clusteringNMI);
fprintf('the running time of Kmeans is %f seconds.\n', runningTime);
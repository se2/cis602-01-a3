clear
clc
close all

%% Global variables
% 
% load dataset
% [trainFea, trainLabel, testFea, testLabel] = LoadData('usps', 10);
[trainFea, trainLabel, testFea, testLabel] = LoadData('pie', 5);
[nFea, n] = size(trainFea);
% 
% polynomial cross validation 5-fold
cross_validation = ' -v 5 ';

%% Linear kernel
% linear with slack C (default 1, 5, 10, 15)
% C = [1, 5, 10, 15];
% train_options = '-t 0 -c ';
% model = [];
% for i=C
%     sub_train_options = [train_options, num2str(i), cross_validation];
%     model(i,:) = svmtrain(trainLabel, trainFea, sub_train_options);
%     clear sub_train_options;
% end
% 
% % get max model with highest accuracy
% [maxValue, maxIndex] = max(model);
% train_options = [train_options, num2str(maxIndex)];
% 
% % start running time
% tic;
% % construct model 
% linearModel = svmtrain(trainLabel, trainFea, train_options);
% % predict using svm
% [predictLabel] = svmpredict(testLabel, testFea, linearModel);
% 
% % stop running time
% runningTime = toc;
%

%% Polynomial kernel
train_options = '-t 1';
gama = [1/nFea, 0.1, 0.001, 1, 10];
degree = [2, 3, 4, 5];
model = [];
for i=gama
    sub_gama = [' -g ', num2str(i)];
    subgama_train_options = [train_options, cross_validation, sub_gama];
    for j=degree
        sub_degree = [' -d ', num2str(j)];
        subdegree_train_options = [subgama_train_options, sub_degree];
        sub_model = svmtrain(trainLabel, trainFea, subdegree_train_options);
        model = [model, vertcat(sub_model, i, j)];
        clear sub_model
        clear subdegree_train_options
        clear sub_degree
    end
    clear subgama_train_options
    clear sub_gama
end
[maxValue, maxIndex] = max(model(1,:));
train_options = [train_options, ' -g ', num2str(model(2,maxIndex)), ' -d ', num2str(model(3,maxIndex))];

% start running time
tic;
% construct model 
polynomialModel = svmtrain(trainLabel, trainFea, train_options);

% predict using svm
[predictLabel] = svmpredict(testLabel, testFea, polynomialModel);

% stop running time
runningTime = toc;
% 

%% Gaussion kernel
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

%% Print result
% 
% compute accuracy
% clusteringAcc = accuracy(testLabel, predictLabel);
% compute the clustering NMI
clusteringNMI = nmi(testLabel, predictLabel);
% 
% fprintf('the clustering accuracy of SVM is %f.\n', clusteringAcc);
% fprintf('the clustering accuracy of SVM(NMI) is %f.\n', clusteringNMI);
fprintf('the running time of Kmeans is %f seconds.\n', runningTime);
% 

%% LoadData function
% 
% source:
%     - usps
%     - pie
% numTrainData:
%     - 10, 50, or 100 for usps
%     - 5, 10, or 15 for pie
function [trainFea, trainLabel, testFea, testLabel] = LoadData(source, numTrainData)
    
    if (strcmpi(source, 'usps'))
        load('USPS.mat');
        % number of samples in each digit
        numSampleEachDigit = 1100;
    elseif (strcmpi(source, 'pie'))
        load('PIE.mat');
        fea = Data;
        gnd = Label;
        clear Data;
        clear Label;
    end
    
    % normalize row feature
    fea = NormalizeFea(fea, 1);

    % get training data and training label
    trainIndex = [];
    testIndex = [];

    if (strcmpi(source, 'usps'))
        for i = 0:9
            trainIndex = [trainIndex, i*numSampleEachDigit + 1: i*numSampleEachDigit + numTrainData];
            testIndex = [testIndex, i*numSampleEachDigit + numTrainData + 1: (1+i)*numSampleEachDigit];
        end
    elseif (strcmpi(source, 'pie'))
        for i = 1:n_per
            trainIndex = [trainIndex, (i-1)*n_sub+1: (i-1)*n_sub+numTrainData];
            testIndex = [testIndex, (i-1)*n_sub+numTrainData+1: i*n_sub];
        end
    end
    
    % return training and testing data
    trainFea = fea(trainIndex,:);
    trainLabel = gnd(trainIndex,:);
    testFea = fea(testIndex,:);
    testLabel = gnd(testIndex,:);
end

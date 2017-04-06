clear
clc
close all

% load dataset
load('USPS.mat');
% normalize row feature
fea = NormalizeFea(fea, 1);

% % reduce demension with PCA
% [eigVector, eigValue] = PCA(fea, 5);
% fea = fea * eigVector;

% number of samples in each digit
numSampleInEachDigit = 1100;

% number of training data
numTrainingInEachDigit = 10;

% get training data and training label
trainIndex = [];
testIndex = [];

for i = 0:9
    trainIndex = [trainIndex, i*numSampleInEachDigit + 1: i* numSampleInEachDigit + numTrainingInEachDigit];
    testIndex = [testIndex, i*numSampleInEachDigit + numTrainingInEachDigit + 1: (1+i)*numSampleInEachDigit];
end


%generate training and testing data
trainFea = fea(trainIndex,:);
trainLabel = gnd(trainIndex,:);
testFea = fea(testIndex,:);
testLabel = gnd(testIndex,:);

% compute model
SVMStruct = fitcecoc(trainFea, trainLabel);



%Preprocessed in Python 
%Random Forest model building and analysis of features

%Load training Data

trainingData=readtable('Train_Preprocessed.csv','PreserveVariableNames',true);
%testData=readtable('Test_Preprocessed.csv');
rng(1); % For reproducibility

inputTable = trainingData;
predictorNames = {'fixed acidity', 'volatile acidity', 'citric acid', 'chlorides', 'total sulfur dioxide', 'density', 'sulphates', 'alcohol'};
predictors = inputTable(:, predictorNames);
response = inputTable.quality;

%xtest=testData(:,1:end-1)
%responseTest=testData.quality;

%% Begin with (Test) 100 Trees 
%rng(1);
%RFM=TreeBagger(100,predictors,response,"OOBPrediction","On",...
  %  'Method','Classification')
%% Create cross-validation for comparison/evaluation

CvTb=cvpartition(response,'KFold',10);

%% Mean Predicition Importance (PI) & building of the model

PI_Mat=[];

for i = 1:CvTb.NumTestSets
% for each fold examine the PI

xtrain = predictors([training(CvTb,i)],:);
ytrain = response([training(CvTb,i)]);

%Build Random forest tree using numPredictors^2=64 trees
RFM=TreeBagger(64,xtrain,ytrain,'Method','classification',...
    'OOBVarImp','On','PredictorSelection','curvature',...
    'OOBPredictorImportance','on');
%Examine importance 

PI=RFM.OOBPermutedPredictorDeltaError;

%Add output to the PI_MAT

PI_Mat=[PI_Mat;PI];

end

%% Mean PI calculation

PI_means=mean(PI_Mat,1)

%% Bar chart showing the mean PI values for all 8 predictors

figure;
bar(PI_means)
title('Importance of each Predictor')
ylabel('Predictor importance estimates')
xlabel('Predictors')

%% Find Scales of features which have the highest importance on the model


idxvar = find(PI_means>1)

% 2,6,7,8 These are the features which have the high

% 'volatile acidity', 'total sulfur dioxide', 'density', 'sulphates', 'alcohol'
%% Evaluate the number of trees and OOB ratio 

finbag = zeros(1,RFM.NumTrees);
for t=1:RFM.NTrees
    
    finbag(t) = sum(all(~RFM.OOBIndices(:,1:t),2));
    
end
finbag = finbag / size(response,1);
figure
plot(finbag)
xlabel('Number of Grown Trees')
ylabel('Fraction of In-Bag Observations')


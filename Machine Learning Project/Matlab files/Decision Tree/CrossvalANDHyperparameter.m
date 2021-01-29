%% Preprocessed in python 

%Load training Data
trainingData=readtable('Train_Preprocessed.csv')
%testData=readtable('Test_Preprocessed.csv')
rng(1); % For reproducibility

inputTable = trainingData;
predictorNames = {'fixedAcidity', 'volatileAcidity', 'citricAcid', 'chlorides', 'totalSulfurDioxide', 'density', 'sulphates', 'alcohol'};
predictors = inputTable(:, predictorNames);
response = inputTable.quality;

%% Partition
c=cvpartition(response,'k',10)

%%  Examine standard tree on data

SDT=fitctree(trainingData,'quality')
view(SDT,'mode','graph');
%%
%calculate Tree initial loss
errTree = loss(SDT,trainingData);
disp("Classification Tree Loss: " + errTree) %0.064343%
%Tree accuracy prior
TreeAccuracy = 1-errTree %0.9357


%Typically, the misclassification error on the training data is 
%not a good estimate of how a model will perform 
%on new data because it can underestimate 
%the misclassification rate on new data.
%A better estimate is the cross-validation error.
%% Cross Validation of training set analysis

%misclassification error and classification accuracy.
%By default, crossval ensures that the class proportions
%in each fold remain approximately the same 
%as the class proportions in the response variable

%building decision tree model
dTree = fitctree(trainingData,'quality','CVPartition',c); %default 10 Fold 'on'

view(dTree.Trained{1},'mode','graph')


%% analysis of cross-validation

 % Performs stratified 10-fold cross-validation
cvtrainError = kfoldLoss(dTree)%0.2127
cvtrainAccuracy = 1-cvtrainError %0.7873

%% Visual representation of Cross-Validation


fold1 = test(c,1)
fold2 = test(c,2)
fold3 = test(c,3)
fold4 = test(c,4)
fold5 = test(c,5)
fold6 = test(c,6)
fold7 = test(c,7)
fold8 = test(c,8)
fold9 = test(c,9)
fold10 = test(c,10)

data = [fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9,fold10];
h = heatmap(double(data),'Colormap',summer);
sorty(h,{'1','2','3','4','5','6','7','8','9','10'},'descend')
xlabel('Repetition')
ylabel('Observation')
title('Test Set Observations')

%% 
view(dTree.Trained{1},'Mode','graph');

%%
%Examine one of the CrossVals
view(dTree.Trained{4},'Mode','graph')
dTree.Trained{4}.NodeSize
dTree.Trained{4}.Children 

%% Calculate the minimum number of leaf size
leafs=logspace(1,2,100);
rng(1)
N = numel(leafs);
err = zeros(N,1);
for n=1:N
    dTreeMinL = fitctree(predictors,response,'CVPartition',c,...
        'MinLeafSize',leafs(n));
    err(n) = kfoldLoss(dTreeMinL);
end
plot(leafs,err);
xlabel('Min Leaf Size');
ylabel('cross-validated error');

%The best leaf size is between about 
%24 and 25 observations per leaf. with high isolation & minimising the
%error

%% Maximum number of splits hyperparameter
iteration =logspace(1,2,100);
rng('default')
N = numel(iteration);
err = zeros(N,1);
for n=1:N
    dTreeMaxSpl = fitctree(predictors,response,'CVPartition',c,...
        'MaxNumSplits',iteration(n));
    err(n) = kfoldLoss(dTreeMaxSpl);
    
    
    
end
plot(iteration,err);
xlabel('Max Num Split');
ylabel('cross-validated error');


%between 11 and 12 splits produced the least validated error

%% minParentSize hyperparameter investigation

iteration =logspace(1,2,100);
rng('default')
N = numel(iteration);
err = zeros(N,1);
for n=1:N
    dTreeminParent = fitctree(predictors,response,'CVPartition',c,...
        'minParentsize',iteration(n));
    err(n) = kfoldLoss(dTreeminParent);
    
    
    
end
plot(iteration,err);
xlabel('MinParentSize');
ylabel('cross-validated error');

% Minparentsize seems to be at optimum between 81 & 86 @ 0.17 error rate
%% Optimisation of tree & results


iteration =logspace(1,1,10);
rng('default')
N = numel(iteration);
err = zeros(N,1);
for n=1:N
    optiMod = fitctree(predictors,response,'CVPartition',c,...
        'minParentsize',83, 'MinLeafSize',24,'MaxNumSplits',11);
    err(n) = kfoldLoss(optiMod);
    
    
end
OptimisationError = kfoldLoss(optiMod)%0.1644
OptimisationAccuracy = 1-OptimisationError %0.8356



%% Loading Tesing DATA
testData=readtable('Test_Preprocessed.csv')
xtest=testData(:,1:end-1)
responseTest=testData.quality;

%% Build Final Optimised Model to be able to test data

FinalDTMdl = fitctree(predictors,response,...
        'minParentsize',83, 'MinLeafSize',24,'MaxNumSplits',11)
    
   
LossFinal = loss(FinalDTMdl,predictors,response)%0.1528
AccuFinal = 1-LossFinal %0.8472

save('Optimised_Decision_Tree_Model.mat','FinalDTMdl') %Save DT for final Testing
    

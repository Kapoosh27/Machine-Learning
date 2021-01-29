%Preprocessed in Python 
%Random Forest model parameter & hyperparameters setting

%Load training Data

trainingData=readtable('Train_Preprocessed.csv','PreserveVariableNames',true);
testData=readtable('Test_Preprocessed.csv');
%Test Data%
xtest=testData(:,1:end-1)
responseTest=testData.quality;
rng(1); % For reproducibility


inputTable = trainingData;
predictorNames = {'fixed acidity', 'volatile acidity', 'citric acid', 'chlorides', 'total sulfur dioxide', 'density', 'sulphates', 'alcohol'};
predictors = inputTable(:, predictorNames);
response = inputTable.quality;
%%
%xtest=testData(:,1:end-1)
responseTest9=categorical(testData.quality);

%%
xtestArray=table2array(testData(:,1:end-1))

%%
responseTestCell=table2cell(testData(:,end))
%% Initialising array to store grid search paramters and results

%Metric=[];
%Predetermined range of trees, minimum leaves and 
%minimum number of predictors
%num_trees=[1 20 40 60 80 100];
%num_leaves=[1 10 20 30 40 50];
%num_predictors=[5 10 15 20 23];
%iteration=0;
%for tree=num_trees
 %  for minleaf=num_leaves      
  %     for numpred=num_predictors
   %        iteration= iteration+1
   TRF = TreeBagger(48,predictors,response,'ClassNames',{'0'  '1'  '2'},'Prior',[0.042 0.822 0.135],'MinLeafSize',1,'Method','classification','NumPredictorsToSample',5);
   
   %% %Initialising array to store grid search paramters and results
Metric=[];
%Predetermined range of trees, minimum leaves and 
%minimum number of predictors
num_trees=[1 20 40 60 80 100];
num_leaves=[1 10 20 30 40 50];
num_predictors=[5 10 15 20 23];
iteration=0;
for tree=num_trees
   for minleaf=num_leaves
       for numpred=num_predictors
           iteration= iteration+1
   treeRandomForest = TreeBagger(tree,predictors,response,'ClassNames',{'0'  '1'  '2'},'Prior',[0.042 0.822 0.135],'MinLeafSize',minleaf,'Method','classification','NumPredictorsToSample',numpred);
   treeRandomForestResult=predict(treeRandomForest,xtestArray);
[predRF, PosteriorRF]=predict(treeRandomForest,xtestArray);
confusion_matrix_RF=confusionmat(responseTest9,categorical(treeRandomForestResult));

%Calculating perfromance metrics corresponding to grid search parameters)
[accuracyRF, fscoreRF, precisionRF, recallRF, specificityRF] = Performance(confusion_matrix_RF);

%Array to store grid search paramters and results
Metric=[Metric;tree minleaf numpred accuracyRF fscoreRF precisionRF recallRF specificityRF;]
       end
   end
end



%[accuracyRF, fscoreRF, precisionRF, recallRF, specificityRF] = Performance(confusion_matrix_RF);

%determining the maximum values for each row to determine the best
%performance metric
maxValues=max(Metric);
optimizedValues=[];

fprintf('Hyperparameters for each objective\n')
fprintf('   trees     minleaf   numpred   accuracyRF   fscoreRF   precisionRF   recallRF  specificityRF\n')
forbestAccuracy=Metric(find(Metric(:,4)==maxValues(4)),:)
fprintf('   trees     minleaf   numpred   accuracyRF   fscoreRF   precisionRF   recallRF  specificityRF\n')
forbestFscore=Metric(find(Metric(:,5)==maxValues(5)),:)
fprintf('   trees     minleaf   numpred   accuracyRF   fscoreRF   precisionRF   recallRF  specificityRF\n')
forbestPrecision=Metric(find(Metric(:,6)==maxValues(6)),:)
fprintf('   trees     minleaf   numpred   accuracyRF   fscoreRF   precisionRF   recallRF  specificityRF\n')
forbestRecall=Metric(find(Metric(:,7)==maxValues(7)),:)
fprintf('   trees     minleaf   numpred   accuracyRF   fscoreRF   precisionRF   recallRF  specificityRF\n')
forbestSpecificity=Metric(find(Metric(:,8)==maxValues(8)),:)

%%

writematrix(Metric,'myDataFile.csv')
%%

function [accuracyRF, fscoreRF, precisionRF, recallRF, specificityRF] = Performance( mat )
% This is a function to compute some performance metrics based on a
% confusion matrix.
% INPUT : Confusion Matrix. Format : (1,1) : TP // (2,2) : TN // (1,2) : FN // (2,1) : FP
% OUTPUT : A vector containing Accuracy, F-Score,
% Precision, Recall and Specificity

accuracyRF = (mat(1,1) + mat(2,2)) / sum(sum(mat));
precisionRF = mat(1,1) / (mat(1,1) + mat(2,1));
recallRF = mat(1,1) / (mat(1,1) + mat(1,2)); % also called sensitivity
specificityRF = mat(2,2) / (mat(2,2) + mat(2,1));
fscoreRF = 2 * precisionRF * recallRF / (precisionRF + recallRF);

end





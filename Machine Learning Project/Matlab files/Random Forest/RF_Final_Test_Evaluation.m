%Preprocessed in Python 
%Random Forest model parameter & hyperparameters setting

%addpath('lossFcn_RF_MCR.m');

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

%% Final Random forest model Train and Test

%%%RFMdlFinal= TreeBagger(60,predictors,response,'OOBPrediction','on',...
   % 'ClassNames',{'0'  '1'  '2'},'Prior',[0.042 0.822 0.135],...
    %'MinLeafSize',1,'Method',...
    %'classification','NumPredictorsToSample',5);
%%
xtestArray=table2array(testData(:,1:end-1))% converting Test predictors to Array
responseTest9=categorical(testData.quality); %converting Test Labels to categories

%RFMdlFinalTest=predict(RFMdlFinal,xtestArray);

%% Load Bayesian Optimised Random Forest model
%for Testing the model below please load my model here by deleting comment
%load('Bayes_Optimised_Random_Forest_Model.mat') %Load RFMdlAutoOpti for prediction and analysis
%% Predict using Bayesian Optimised Random Forest model & examine results
RFMdlFinalResultAuto=predict(RFMdlAutoOpti,xtestArray);

[predRF, PosteriorRF]=predict(RFMdlAutoOpti,xtestArray);
%confusion matrix for random forest classifier
figure(1)
confusionchart(responseTest9,categorical(RFMdlFinalResultAuto));
confusion_matrix_RF=confusionmat(responseTest9,categorical(RFMdlFinalResultAuto));
%calculating the perfromance metrics for random forest classifier
[accuracyRF,precisionRF, recallRF, specificityRF,fscoreRF] = Performance(confusion_matrix_RF);
fprintf('Performance Metrics for Random Forest\n')
fprintf('Accuracy RF : %f\n',accuracyRF)
fprintf('Precision RF : %f\n',precisionRF)
fprintf('Recall RF : %f\n',recallRF)
fprintf('Specificity RF : %f\n',specificityRF)
fprintf('F1 score RF : %f\n',fscoreRF)

title('Confusion Matrix for Random Forest (Auto) Classifier')
%% Final Training using Hyper parameters found in the GridSearch
%Final Test used to examine results below
%%% RFMdlFinal = TreeBagger(60,predictors,response,'ClassNames',{'0'  '1'  '2'},'Prior',[0.042 0.822 0.135],'MinLeafSize',1,'Method','classification','NumPredictorsToSample',5);

%for Testing the model below please load my model here by deleting comment
 %load('RF_Optimised_model_manual_hyperparameters.mat')

%% TEST Results Manual gridsearch Model

RFMdlFinalResult=predict(RFMdlFinal,xtestArray); %Test

[predRF, PosteriorRF]=predict(RFMdlFinal,xtestArray);
%confusion matrix for random forest classifier
figure(1)
confusionchart(responseTest9,categorical(RFMdlFinalResult));
confusion_matrix_RF=confusionmat(responseTest9,categorical(RFMdlFinalResult));
%calculating the perfromance metrics for random forest classifier
[accuracyRF,precisionRF, recallRF, specificityRF,fscoreRF] = Performance(confusion_matrix_RF);
fprintf('Performance Metrics for Random Forest\n')
fprintf('Accuracy RF : %f\n',accuracyRF)
fprintf('Precision RF : %f\n',precisionRF)
fprintf('Recall RF : %f\n',recallRF)
fprintf('Specificity RF : %f\n',specificityRF)
fprintf('F1 score RF : %f\n',fscoreRF)

title('Confusion Matrix for Random Forest (Manual) Classifier')
%%

save('RF_Optimised_model_manual_hyperparameters.mat','RFMdlFinal') %Save Model for crossEvaluation
%%

function [accuracyRF, fscoreRF, precisionRF, recallRF, specificityRF] = Performance( mat )
% This is a function to compute some performance metrics based on a
% confusion matrix.
% INPUT : Confusion Matrix. Format : (1,1) : TP // (2,2) : TN // (1,2) : FN // (2,1) : FP
% OUTPUT : A vector containing Accuracy, Geometric Mean, F-Score,
% Precision, Recall and Specificity

accuracyRF = (mat(1,1) + mat(2,2)) / sum(sum(mat));
precisionRF = mat(1,1) / (mat(1,1) + mat(2,1));
recallRF = mat(1,1) / (mat(1,1) + mat(1,2)); % also called sensitivity
specificityRF = mat(2,2) / (mat(2,2) + mat(2,1));
fscoreRF = 2 * precisionRF * recallRF / (precisionRF + recallRF);

end


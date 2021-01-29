
%% Loading Testing DATA
testData=readtable('Test_Preprocessed.csv')
xtest=testData(:,1:end-1)
responseTest=testData.quality;
%% Load Optimised DT evaluating hyperparameters

load('Optimised_Decision_Tree_Model.mat')

%% Prediction of Partitioned Variables
FinalPrediction = predict(FinalDTMdl,xtest)
%% Evaluation of Loss/Accuracy

FinalLoss = loss(FinalDTMdl,xtest,responseTest)%0.1761
FinalAccuracy = 1-FinalLoss %0.8239

%% Evaluation 

% Initally we found the decision tree to be robust and had a high Error
% rate, where the accuracy was at 78% when model was yet to be trained; we
% found the highest accuracy being on the training data when being cross
% validated we had an accuracy of 83% although this was averaged I wanted
% to see how this model would do if it was to predict on the Training data;
% Out come showed an increase of 1.7%, I had doubts that potentially the
% model may have been overfitted onto the data however after examining
% testing the model on new data which the model has yet to see we find the
% accuracy dropping by 2.33% which suggests although model had not been overfitted

%Final training of the Decision Tree Model
%2.33% dip

%Final test show a slight dip at 82% accuracy which is still an improvement
%to the 78% initial results



%% Manual Confusion matrix

cm=confusionchart(responseTest,FinalPrediction)
cm.NormalizedValues
cm.Title='Red wine classification using Manual GridSearch Decision Tree Matrix ';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized'

%% Evaluate Performance

%FinalPrediction = predict(FinalDTMdl,xtest); %Test

%[predRF, PosteriorRF]=predict(FinalDTMdl,xtest);
%confusion matrix for random forest classifier
%figure(1)
%confusionchart(responseTest,FinalPrediction);
%confusion_matrix_DT=confusionmat(responseTest,FinalPrediction);

%calculating the perfromance metrics for Decision Tree Manual Optimisation classifier
%[accuracyDT,precisionDT, recallDT, specificityDT,fscoreDT] = Performance(confusion_matrix_DT);
%fprintf('Performance Metrics for Random Forest\n')
%fprintf('Accuracy DT : %f\n',accuracyDT)
%fprintf('Precision DT : %f\n',precisionDT)
%fprintf('Recall DT : %f\n',recallDT)
%fprintf('Specificity DT : %f\n',specificityDT)
%fprintf('F1 score DT : %f\n',fscoreDT)

%title('Confusion Matrix for Decision Tree Manual Optimisation Classifier')


%% Seems like I need to Evaluate Test Data
[C,ia,ic]=unique(responseTest)
a_counts = accumarray(ic,1);
value_counts = [C, a_counts]

% 15(0),399(1),66(2)
% 3.125% 83.125% 13.75%
% This therefore shows the was an error in my inital preprocessed DAta
% which I will evaluate in my poster


%% Comparing my hyperparameter with the 'auto' optimisation approach
x=table2array(trainingData(:,1:end-1))
y=table2array(trainingData(:,end))


AutoOptiModel = fitctree(x,y,'OptimizeHyperparameters','auto')
%From the optimizehyperparameter function we find that the best 'minleaf is
%at 29 which produced a 0.16543 Error rate however we have produced a
%better model considering I also considered a greater scope of the model
%also scrutinising minParentsize aswell as MaxNumSplits

%Best estimated feasible point (according to models):
 %   MinLeafSize
  %  ___________

   %     22    

%% Save Model for crossEvaluation
save('AUTO_Optimised_Decision_Tree_Model.mat','AutoOptiModel') %Save Model for crossEvaluation

%% Auto Confusion Tree

cm=confusionchart(responseTest,FinalPredictionAuto)
cm.NormalizedValues
cm.Title='Red wine classification using Bayesian Optimisation Decision Tree Matrix';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized'

%% Load Auto Optimised
load('AUTO_Optimised_Decision_Tree_Model.mat')

%% Convert test predictors to array
xtestArray=table2array(testData(:,1:end-1))
%%

FinalPredictionAuto = predict(AutoOptiModel,xtestArray)
%% Evaluation of Loss/Accuracy

FinalLossAuto = loss(AutoOptiModel,xtestArray,responseTest)%0.1679
FinalAccuracyAuto = 1-FinalLossAuto %0.8321

%1% difference hence very similar Accuracy hence we can confirm minimal disagreement in hyperparameter
%method something I may consider when building my second model

%% Auto Performance

%FinalPredictionAuto = predict(AutoOptiModel,xtestArray); %Test

%[predRF, PosteriorRF]=predict(AutoOptiModel,xtestArray);
%confusion matrix for random forest classifier
%figure(1)
%confusionchart(responseTest,FinalPredictionAuto);
%confusion_matrix_DT=confusionmat(responseTest,FinalPredictionAuto);

%calculating the perfromance metrics for Decision Tree Manual Optimisation classifier
%[accuracyDT,precisionDT, recallDT, specificityDT,fscoreDT] = Performance(confusion_matrix_DT);
%fprintf('Performance Metrics for Decision Tree\n')
%fprintf('Accuracy DT : %f\n',accuracyDT)
%fprintf('Precision DT : %f\n',precisionDT)
%fprintf('Recall DT : %f\n',recallDT)
%fprintf('Specificity DT : %f\n',specificityDT)
%fprintf('F1 score DT : %f\n',fscoreDT)

%title('Confusion Matrix for Decision Tree Auto Optimisation Classifier')


%% Visualise Decision Trees

%Optimisation of Manual hyperparameters 
view(FinalDTMdl,'Mode','graph')

%Optimisation of auto hyperparameters 
view(AutoOptiModel,'Mode','graph')


%% Grid Search Performance Metrics For Decision Tree

%function [accuracyDT, fscoreDT, precisionDT, recallDT, specificityDT] = Performance( mat )
% This is a function to compute some performance metrics based on a
% confusion matrix.
% INPUT : Confusion Matrix. Format : (1,1) : TP // (2,2) : TN // (1,2) : FN // (2,1) : FP
% OUTPUT : A vector containing Accuracy, Geometric Mean, F-Score,
% Precision, Recall and Specificity

%accuracyDT = (mat(1,1) + mat(2,2)) / sum(sum(mat));
%precisionDT = mat(1,1) / (mat(1,1) + mat(2,1));
%recallDT = mat(1,1) / (mat(1,1) + mat(1,2)); % also called sensitivity
%specificityDT = mat(2,2) / (mat(2,2) + mat(2,1));
%fscoreDT = 2 * precisionDT * recallDT / (precisionDT + recallDT);

%end








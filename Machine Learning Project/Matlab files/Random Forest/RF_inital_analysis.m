%Preprocessed in Python 
%Random Forest model building and analysis

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

%% Build random forest trees

Mdl=TreeBagger(64,predictors,response,'OOBPrediction','On',...
    'OOBPredictorImportance','On')
%Chose to begining my Ensembled Decision tree model with 64 trees as an
%arbitary choice as I have 8 predicting categories I used 8^2 

view(Mdl.Trees{1},'Mode','graph')
%view First tree built
view(Mdl.Trees{64},'Mode','graph')
%view Last tree built

%% Evaluating the initial 
figure
plot(oobError(Mdl))
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Classification Error')

%This graph shows how the ensemble error changes with accumulation of
%trees we find that the error is at a peak at 4 trees at 22.43% whereas we
%see a steep decine in error as more trees are built with the lowest point
%of error recorded at 15.46%

%% examination of the proportion of data are Out-In the model testing 

finbag = zeros(1,Mdl.NumTrees);
for t=1:Mdl.NTrees
    finbag(t) = sum(all(~Mdl.OOBIndices(:,1:t),2));
end
finbag = finbag / size(predictors,1);
figure
plot(finbag)
xlabel('Number of Grown Trees')
ylabel('Fraction of In-Bag Observations')


% we examined that the optimal where all the data is in bag and used for
% estimation is between 11-12 trees

%% Feature importance 

figure
bar(Mdl.OOBPermutedPredictorDeltaError)
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Feature Importance')

%Just like the decision tree model and the preprocessing has shown both the
%volatile acidity and Alcohol ha the highest feature imporance which shows
%the high correlation plays a part

% I will take 1 as my arbitrary value where everything above this point has
% a relatively high yield on the estimation of the response value
% throughout the trees

%%

idxvar = find(Mdl.OOBPermutedPredictorDeltaError>1)

% we find that there are 5 categories above 1 which are [2,3,6,7,8]
% translates to ('volatile acidity', 'citric acid', 'density', 'sulphates',
% 'alcohol')

%% evaluate the Random Forrest with only these features

Mdl5=TreeBagger(64,predictors(:,idxvar),response,'OOBPredictorImportance','off',...
    'OOBPrediction','on');
figure
plot(oobError(Mdl5))
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Classification Error')


% using only the top 5 features we find that although it isosilate slight
% more taking it longer to reach the minimum classification rate evaluate
% over 64 trees; reaching 0.1492 at 55 trees where as our initial model
% reached its lowest error rate at 0.1466 at the 61 trees 

%% Evaluating the margin score changes per-tree 
figure
plot(oobMeanMargin(Mdl5));
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Mean Classification Margin')


figure
plot(oobMeanMargin(Mdl));
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Mean Classification Margin')

%% Matrix of proximities and examine the distribution of outlier measures

Mdl = fillProximities(Mdl);
figure
histogram(Mdl.OutlierMeasure)
xlabel('Outlier Measure')
ylabel('Number of Observations')



Mdl5 = fillProximities(Mdl5);
figure
histogram(Mdl5.OutlierMeasure)
xlabel('Outlier Measure')
ylabel('Number of Observations')



%% Find class of extreme outliers

extremeOutliers = Mdl5.Y(Mdl5.OutlierMeasure>10)

percentAvg = 100*sum(strcmp(extremeOutliers,'1'))/numel(extremeOutliers)

%we see here that 100% of the outliers are marked as average wine quality 
% looking back at the decision matrix we find that the majority of
% misclassifications came from the average rating
%%

AvgPosition = find(strcmp('1',Mdl5.ClassNames))


%% ROC curve examining the rate at which the AvgPosition is predicted correctly 

[Yfit,Sfit] = oobPredict(Mdl5);

[fpr,tpr] = perfcurve(Mdl5.Y,Sfit(:,AvgPosition),'1');
figure
plot(fpr,tpr)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
%% Ensemble Accuracy vs Threshold on average rating of red wine 
[fpr,accu,thre] = perfcurve(Mdl5.Y,Sfit(:,AvgPosition),'1','YCrit','Accu');
figure(20)
plot(thre,accu)
xlabel('Threshold for ''Average Rating'' Returns')
ylabel('Classification Accuracy')

%% Accuracy using default 0.5 threshold
accu(abs(thre-0.5)<eps) %0.8418-0.8409
%% Maximum Accuracy 

[maxaccu,iaccu] = max(accu) %0.8615

%Therefore 

%Optimal Threshold is at 

thre(iaccu) %0.3636 Threshold 






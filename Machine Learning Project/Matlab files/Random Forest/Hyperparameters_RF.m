%Preprocessed in Python 
%Random Forest model parameter & hyperparameters setting

%utilising the Bayesian Optimisation here we look at how we could optimise
%hyperparameters using and auto in built function however my 'Gridsearch'
%method is where I evaluated Hyperparameters manually is what I used for my
%final Model for RF

%Load training Data

trainingData=readtable('Train_Preprocessed.csv','PreserveVariableNames',true);
%testData=readtable('Test_Preprocessed.csv');
%Test Data%
%xtest=testData(:,1:end-1)
%responseTest=testData.quality;
rng(1); % For reproducibility


inputTable = trainingData;
predictorNames = {'fixed acidity', 'volatile acidity', 'citric acid', 'chlorides', 'total sulfur dioxide', 'density', 'sulphates', 'alcohol'};
predictors = inputTable(:, predictorNames);
response = inputTable.quality;

%xtest=testData(:,1:end-1)
%responseTest=testData.quality;

%% Create cross-validation for comparison/evaluation

CvTb=cvpartition(response,'KFold',10);

%% Building of Model from feature tab analysis
RFMD=TreeBagger(400,predictors,response,...
    'OOBPredictorImportance','on');

% I will be exploring 3 parameters and examing hyperparameters of the
% 'MinLeafSize','NumPredictorsToSample','NumTrees'
%Examing documentation on Matlab it seems the main route in analysis
%%
oobErrorBaggedEnsemble = oobError(RFMD)

%% Visualise OOB error Ensemble 

figure(1);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';
title('Out-of-bag classification error v/s Number of trees')
color='red'
%Lowest point reached at 157 Trees where OOB Error at 0.1448% however an
%earlier error is reached at 35 trees at 0.1466% 

savefig('Visual_OOB_400Trees_SAMPLE_HP_RF')

%% Lets begin by selecting some parameters to examine 
%lets look at some leaf ranges
leaf = [1 10 20 50 80 100];

%Examining previous visualisation of OOB Error we find the optimal number of trees
%to minimise error to be at 157 hence I will select a range of 160 Trees
nTrees=160;
rng=1 
color='bgr';

for leaves = 1:length(leaf)
   %Allow random samples to be generated without leaving the leaf
   %sample stated above
   
   %bagged decision tree for each leaf size and plot out-of-bag
   % error 'oobError'
   RFSample = TreeBagger(nTrees,predictors,response,'OOBPrediction','on',...
                         'MinLeafSize',leaf(leaves));
   plot(oobError(RFSample))
   hold on
end

xlabel('Number of grown trees')
ylabel('Out-of-bag classification error')
legend({'1', '10', '20','50','80','100'},'Location','NorthEast')
title('Classification Error for Different Leaf Sizes')
hold off

%savefig('Visual_OOB_160TRees_Loop_Vary_Leaf_Size_HP_RF')

% Suprisingly a minimum Trees sample of 48 Trees and Only 1 leaf supplied
% the lowest level of error at 0.1439% Error rate 

%% Examining out predictors
% number of predictors
numPred = [1 2 3 4 5 6 7 8];

%Examining previous visualisation of OOB Error we find the optimal number of trees
%to minimise error to be at 157 hence I will select a range of 160 Trees
nTrees=160;
rng=1 
color='bgr';

for X = 1:length(numPred)
   %Allow random samples to be generated without leaving the leaf
   %sample stated above
   
   %bagged decision tree for each leaf size and plot out-of-bag
   % error 'oobError'
   RFSample = TreeBagger(nTrees,predictors,response,'OOBPrediction','on',...
                         'MinLeafSize',1,'NumPredictorsToSample',numPred(X));
   plot(oobError(RFSample))
   hold on
end

xlabel('Number of grown trees')
ylabel('Out-of-bag classification error')
legend({'1', '2','3','4','5','6','7','8'},'Location','NorthEast')
title('Classification Error for Different number of predictors')
hold off

%%savefig('Visual_OOB_160TRees_Loop_Vary_PredTOSAMple_Size_HP_RF')

% We find that the lowest Error rate at 13.58% was seen using 5 predictors
% predictors considering at the optimal 48 Trees we find the Error rate for
% using all 5 classification is at 0.1412

% therefore we will now look into improtance of the features and see which
% 5 could use for the final model

%% Feature importance
%From the initial RF investigation we found the top 5 Features

figure
bar(RFMD.OOBPermutedPredictorDeltaError)
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Feature Importance')

%Features 2,3 and 6,7,8 have the most influence on the data upon

%Just like the decision tree model and the preprocessing has shown both the
%volatile acidity and Alcohol ha the highest feature imporance which shows
%the high correlation plays a part

%considering the Fluxuation of data discrepency is below 2% I feel negating more 
%features would take away the too much impurity in the model hence giving other
%feature much more dominance in prediction

%%%savefig('Visual_Feature_Importance_HP_RF')
%% Specify Tuning Parameters
% Min No of Observations per leaf
%The complexity (depth) of the trees in the forest.
%Deep trees tend to over-fit, but shallow trees tend to underfit.

%maxMinLS = 50;
%minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');


% Remove predictor variables which have a mean Predictor 
% Importance < 1 based 2,6,7,8 
%which are 'volatile acidity', 'total sulfur dioxide', 'density', 'sulphates', 'alcohol'

In_high_imp_variables = removevars(predictors,{'fixed acidity','chlorides','citric acid','total sulfur dioxide'});

% number of predictors at each spliting of node
%numPTS = optimizableVariable('numPTS',[1,size(In_high_imp_variables,2)],'Type','integer'); 

%Function that implements Bayesian optimization
% Hyperparamter object to be inputted into bayesopt function
%hyperparametersRF = [minLS; numPTS; numTrees];

%% Hyperparameter initial observation

minLS = optimizableVariable('minLS',[1,20],'Type','integer');
numPTS = optimizableVariable('numPTS',[1,100],'Type','integer');
% numTrees= optimizableVariable('numTrees',[1,500],'Type','integer');
hyperparametersRF = [minLS;numPTS];
rng(1);
fun = makeFun(In_high_imp_variables, response);
results = bayesopt(fun,hyperparametersRF);

besthyperparameters = bestPoint(results);

%savefig('Visual_Auto_HP_RF')


%% BEst MinLS and numPTS
zbest=bestPoint(results)

%% building optimised Random Forest Model trees with Hyperparameters found using Bayesian Optimisation

RFMdlAutoOpti=TreeBagger(160,predictors,response,'method','classification','OOBPrediction','on',...
            'MinLeafSize',19,'NumPredictorstoSample',8);

save('Bayes_Optimised_Random_Forest_Model.mat','RFMdlAutoOpti') %Save RF for final Testing

%% Function Optimisation 
function fun = makeFun(In_high_imp_variables, response)
%pass to bayesopt
fun = @f;
    % A nested function that uses X and Y
    function oobMCR = f(hparams)
        opts=statset('UseParallel',true);
        numTrees=395;
        A=TreeBagger(numTrees,In_high_imp_variables,response,'method','classification','OOBPrediction','on','Options',opts,...
            'MinLeafSize',hparams.minLS,'NumPredictorstoSample',hparams.numPTS);
        oobMCR = oobError(A, 'Mode','ensemble');
    end
end
%250 Trees
%Observed objective function value = 0.15825
%Estimated objective function value = 0.15734
%Function evaluation time = 2.1382
%Best estimated feasible point (according to models):
 %   minLS    numPTS
  %  _____    ______

   %   5        1   

%500 Trees
 %minLS    numPTS
  %  _____    ______

   %   4        65  

%Estimated objective function value = 0.15175
%Estimated function evaluation time = 4.026
%600 Trees
%Best observed feasible point:
 %   minLS    numPTS
  %  _____    ______

   %   4        41  

%Observed objective function value = 0.15371
%Estimated objective function value = 0.15495
%Function evaluation time = 5.1905

%160 Trees
%Best observed feasible point:
 %   minLS    numPTS
  %  _____    ______

   %  19        8  

%Observed objective function value = 0.15192
%Estimated objective function value = 0.15652
%Function evaluation time = 1.1129

%210

%Best observed feasible point:
 %   minLS    numPTS
  %  _____    ______

   %   6        1   

%Observed objective function value = 0.15103
%Estimated objective function value = 0.15297
%Function evaluation time = 1.366


%64 Trees
%Best estimated feasible point (according to models):
 %   minLS    numPTS
  %  _____    ______

   %   1        2   

%Estimated objective function value = 0.1553
%Estimated function evaluation time = 0.93234

%395 Trees
%Best estimated feasible point (according to models):
%    minLS    numPTS
 %   _____    ______

  %    7        1   

%Estimated objective function value = 0.15725
%Estimated function evaluation time = 4.6467
%%


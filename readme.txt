
methods with target recall = 0.95, 0.99, None (= symmetric misclassification costs)

AdaCOS / COS/ fullModel:
1.
precalculateAllGreedyFeatureSets.py
precalculateAllNonLinearL1FeatureSets.py


2.
getOptimalSequence_recall.py
getOptimalSequence_totalCosts.py


3.
Evaluation:
showAllResults.py
showAllResults_cmpL1andGreedy.py
showAllResults_recallOnly.py



****************************************************************

for missing value imputation use
"missingvalue_imputation.py"

imputationMethod = "gaussian_imputation" 
was used for 
"pyhsioNetWithMissing_5foldCV"
"heartDiseaseWithMissing_5foldCV"


****************************************************************

running Shim2018 baseline

1. run main.py in dynamicCovariateBaselines/Joint-AFA-Classification_modified, e.g.:
~/mantogpu3_environment/intelpython3/bin/python main.py --data_type=heartDiseaseWithMissing_5foldCV --n_envs=50 --foldId 4
~/mantogpu3_environment/intelpython3/bin/python main.py --data_type=pima_5foldCV --foldId 0

2. run:
prepareAllResults_Shim2018.py

3. run:
showAllResults.py

****************************************************************

running GreedyMiser baseline
-> normalize the scores to get probabilities 

1. run:
prepareForMatlabBaselines.py

2. change to matlab (manto38: /data/k-sasaki/Mathwokrs/R2014b/bin) and open folder:
dynamicCovariateBaselines/GreedyMiser

3.in matlab, set "finalTraining=false" in myRun.m and run:
myRun()


------ for target recall
5. find best hyperparameters using:
analyzeForGreedyMiser_targetRecall.py
after setting targetRecall

3.in matlab in file myRun.m set
finalTraining=true
and 
COST_TYPE_STR = '0.95targetRecall' or COST_TYPE_STR = '0.99targetRecall' 
and run:
myRun()

5. prepare results for python using:
analyzeForGreedyMiser_targetRecall.py

------ for asymmetricCost
5. find best hyperparameters using:
analyzeForGreedyMiser_targetRecall.py

3.in matlab in file myRun.m set
finalTraining=true
and 
COST_TYPE_STR = 'asymmetric'
and run:
myRun()

5. prepare results for python using:
analyzeForGreedyMiser_asymmetric.py
 

6. investigate results using:
showAllResults.py


****************************************************************

running AdaptGBRT baseline
-> not used for targetRecall since no probabilities can be acquired (difficult to acquire probabilities since high quality model and low quality model are different)  

1. run:
prepareForMatlabBaselines.py

2. change to matlab (manto38: /data/k-sasaki/Mathwokrs/R2014b/bin) and open folder:
dynamicCovariateBaselines/AdaptApprox-master/ADAPT_GBRT

3. set dataName in myRunAdaptGBRT.m

4. run in matlab:
myRunAdaptGBRT(0)
myRunAdaptGBRT(1)
myRunAdaptGBRT(2)
myRunAdaptGBRT(3)
myRunAdaptGBRT(4)

5. prepare results for python using:
analyzeForAdaptGBRT_totalCosts.py

6. investigate results using:
showAllResults.py



****************************************************************

physioNet Data:

preparePhysioNetData.py was used to transform the whole data into a numpy matrix with number of samples = 12000 and number of variables = 42

analyzePhysioNetData.py:
getting the number of samples where all variables of a certain set are available.

physioNet.py:
explanation of all 42 variables

preparePyhsioNetDataWithMissingAttributes() in realdata.py:
5 fold splitting of noMissingData for test, all others are always used for training


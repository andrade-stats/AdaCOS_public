import numpy
import realdata
import sklearn.linear_model
import sklearn.metrics
import evaluation
import experimentSetting
import experimentHelper
import prepareFeatureSets
import sklearn.gaussian_process
import time
import constants
import preprocessing
import pickle
import pygam
from pygam import LogisticGAM
import sys


# /opt/intel/intelpython3/bin/python evalClassificationAccuracyOnly_onlyGAM.py breastcancer_5foldCV 10
# /opt/intel/intelpython3/bin/python evalClassificationAccuracyOnly_onlyGAM.py pyhsioNetWithMissing_5foldCV 10
# /opt/intel/intelpython3/bin/python evalClassificationAccuracyOnly_onlyGAM.py heartDiseaseWithMissing_5foldCV 10
# /opt/intel/intelpython3/bin/python evalClassificationAccuracyOnly_onlyGAM.py pima_5foldCV 10

# MODEL_TYPE = "logReg" 
MODEL_TYPE = "Combined"
# MODEL_TYPE = "GAM"

if len(sys.argv) == 1:
    dataName = "pima_5foldCV"
    # dataName = "breastcancer_5foldCV"
    # dataName = "heartDiseaseWithMissing_5foldCV"
    # dataName = "pyhsioNetWithMissing_5foldCV"
    
    pyGamValues = 10

else:
    dataName = sys.argv[1]
    pyGamValues = int(sys.argv[2])





if dataName == "pima_5foldCV":
    falsePositiveCost = 800 
    falseNegativeCost = falsePositiveCost
    correctClassificationCost = -50.0 # in order to align to the setting in (Ji and Carin, 2007; Dulac-Arnold et al., 2012) for Diabetes.        
    misclassificationCosts = numpy.zeros((2, 2))
    misclassificationCosts[0, 1] = falsePositiveCost 
    misclassificationCosts[1, 0] = falseNegativeCost 
    misclassificationCosts[0, 0] = correctClassificationCost
    misclassificationCosts[1, 1] = correctClassificationCost
else:
    misclassificationCosts = None


# pima 
# UBRE: AUC = 0.8318 (0.019)
# AIC: AUC = 0.8271 (0.0249)
# AICc: AUC = 0.8287 (0.0267)

# breastcancer
# UBRE: AUC = 0.9838 (0.0136)
# AIC: AUC = 0.9743 (0.0087)
# AICc: AUC = 0.9743 (0.0087)

# heartDisease
# UBRE: AUC = 0.9035 (0.0199)
# AIC: AUC = 0.8902 (0.0422)
# AICc: AUC = 0.9103 (0.0191)


imputationMethod = "gaussian_imputation"

# /opt/intel/intelpython3/bin/python evalClassificationAccuracyOnly_onlyGAM.py

# allTargetRecalls = [0.95, 0.99, 0.999]
allTargetRecalls = [0.95]

NUMBER_OF_FOLDS = 5

# definedFeatureCosts = realdata.getFeaturesCosts(dataName)
# testAccuracyAllFoldsSVM = numpy.zeros(NUMBER_OF_FOLDS)

# testAccuracyAllFoldsLogistic = numpy.zeros(NUMBER_OF_FOLDS)
testAUCAllFoldsLogistic = numpy.zeros(NUMBER_OF_FOLDS)
testAccuracyAllFoldsLogistic = numpy.zeros(NUMBER_OF_FOLDS)
testMisclassificationCosts = numpy.zeros(NUMBER_OF_FOLDS)

# testAccuracyAllFoldsGP = numpy.zeros(NUMBER_OF_FOLDS)
testAUCAllFoldsGP = numpy.zeros(NUMBER_OF_FOLDS)

testRecallAllFoldsLogistic = []
testSpecifityAllFoldsLogistic = []
testFDRAllFoldsLogistic = []
testRecallAllFoldsLogistic_exactRecall = []
testFDRAllFoldsLogistic_exactRecall = []
for i in range(len(allTargetRecalls)): 
    testRecallAllFoldsLogistic.append(numpy.zeros(NUMBER_OF_FOLDS))
    testSpecifityAllFoldsLogistic.append(numpy.zeros(NUMBER_OF_FOLDS))
    testFDRAllFoldsLogistic.append(numpy.zeros(NUMBER_OF_FOLDS))
    testRecallAllFoldsLogistic_exactRecall.append(numpy.zeros(NUMBER_OF_FOLDS))
    testFDRAllFoldsLogistic_exactRecall.append(numpy.zeros(NUMBER_OF_FOLDS))

bestModelIsGAM = numpy.zeros(NUMBER_OF_FOLDS)


for foldId in range(NUMBER_OF_FOLDS):
    
    print("process fold = ", foldId)
    
    trainData, trainLabels, unlabeledData, testData, testLabels = realdata.loadSubset(dataName, None, foldId, imputationMethod)
        
    
    print("run LogisticGAM")
    
    # finalModel, predictedProbsBestModel_trainData = evaluation.getBestModel(trainData, trainLabels, MODEL_TYPE, None)
    
    finalModel, score = evaluation.getBestGAM_AIC(trainData, trainLabels)
    
    # trainedModel.gridsearch(trainData, trainLabels, lam = numpy.logspace(-3, 5, pyGamValues), objective='UBRE', keep_best=True)
    
    bestModelIsGAM[foldId] = isinstance(finalModel, pygam.pygam.LogisticGAM)
    
     
#     for i in range(len(allTargetRecalls)):
#         threshold = evaluation.getThresholdFromPredictedProbabilities(trainLabels, predictedProbsBestModel_trainData, allTargetRecalls[i])
#         testRecallAllFoldsLogistic[i][foldId], testSpecifityAllFoldsLogistic[i][foldId], testFDRAllFoldsLogistic[i][foldId] = evaluation.getTestRecallSpecifityFDR(finalModel, testData, testLabels, threshold)
#         testRecallAllFoldsLogistic_exactRecall[i][foldId], testFDRAllFoldsLogistic_exactRecall[i][foldId] = evaluation.getFDR_atExactRecall(finalModel, testData, testLabels, allTargetRecalls[i])
    
    testAUCAllFoldsLogistic[foldId] = evaluation.getTestAUC(finalModel, testData, testLabels)
    
    testAccuracyAllFoldsLogistic[foldId] = evaluation.getTestAccuracy(finalModel, testData, testLabels)

#     if misclassificationCosts is not None:
#         predictedProbTrueLabel = evaluation.getPredictedProb(finalModel, testData)
#         predictedLabels = evaluation.bayesClassifier(predictedProbTrueLabel, misclassificationCosts)
#         testMisclassificationCosts[foldId] = evaluation.getAverageMisclassificationCosts(testLabels, predictedLabels, misclassificationCosts)


print("")
print("MODEL_TYPE = ", MODEL_TYPE)
print("BEST MODEL IS GAM")
print(bestModelIsGAM)

print("")
print("AVERAGE RESULTS 5-FOLD CROSS-VAlIDATION")

print("*************************** AVERAGE ACCURACY OVER ALL FOLDS  *******************************************")
print("dataName = ", dataName)
print("pyGamValues = ", pyGamValues)

# print("*************************** GAM *******************************************")
# for i in range(len(allTargetRecalls)):
#     print("target recall = ", allTargetRecalls[i])
#     evaluation.showHelperDetailed("    recall = ", testRecallAllFoldsLogistic[i])
#     evaluation.showHelperDetailed("    FDR = ", testFDRAllFoldsLogistic[i])
#     evaluation.showHelperDetailed("    specifity = ", testSpecifityAllFoldsLogistic[i])
#     print("--- results at exact target recall: ---- ")
#     evaluation.showHelperDetailed("    recall = ", testRecallAllFoldsLogistic_exactRecall[i])
#     evaluation.showHelperDetailed("    FDR = ", testFDRAllFoldsLogistic_exactRecall[i])

evaluation.showHelperDetailed("Misclassification Costs = ", testMisclassificationCosts)    
evaluation.showHelperDetailed("Accuracy = ", testAccuracyAllFoldsLogistic)
evaluation.showHelperDetailed("AUC = ", testAUCAllFoldsLogistic)


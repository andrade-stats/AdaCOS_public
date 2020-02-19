
import numpy
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.svm
import scipy.io
import experimentSetting
import experimentHelper
import preprocessing
import sklearn.gaussian_process
from pygam import LogisticGAM
from copy import deepcopy

import constants

# checked
def getWeightedAccuracy(trueLabelRatio, falsePositiveCost, avgMisclassificationCost):
    assert(avgMisclassificationCost >= 0.0)
    avgMaximalMisclassifcationCost = falsePositiveCost * (trueLabelRatio * constants.FN_TO_FP_RATIO + (1.0 - trueLabelRatio))
    assert(avgMisclassificationCost <= avgMaximalMisclassifcationCost)
    return (avgMaximalMisclassifcationCost - avgMisclassificationCost) / avgMaximalMisclassifcationCost
    

def getAllWeightedAccuracyies(trueLabelRatio, allMisClassificationCosts):
    weightedAccs = numpy.zeros(len(constants.allFalsePositiveCosts))

    for i in range(len(constants.allFalsePositiveCosts)):
        weightedAccs[i] = getWeightedAccuracy(trueLabelRatio, constants.allFalsePositiveCosts[i], allMisClassificationCosts[i])
    
    return weightedAccs

# used to set baselines to same recall as proposed method
def getTargetRecallFromProposedMethod(dataName, falsePositiveCost, targetRecall):
    proposedMethodName = "getOptimalSequence_dynamic_BR_noUnlabeledData_" + "nonLinearL1" + "_" + "Combined"
    allTotalCosts, allFeatureCosts, allMisClassificationCosts, allAccuracies, allAUC, allRecall, allFDR, allOperationCosts, allRecall_atExactRecall, allFDR_atExactRecall, allOperationCosts_atExactRecall = experimentHelper.ResultsRecorder.readResults(dataName + "_" + proposedMethodName + "_" + str(targetRecall) + "recall")
    
    correspondingRecallRow =  allRecall[allRecall[:,0] == falsePositiveCost]
    assert(correspondingRecallRow.shape[0] == 1 and correspondingRecallRow.shape[1] == 3)
    avgRecallProposedMethod = correspondingRecallRow[0,1]
    
    # print("allRecall = ")
    # print(allRecall)
    # print("correspondingRecallRow = ")
    # print(correspondingRecallRow)
    # print("avgRecallProposedMethod = ")
    # print(avgRecallProposedMethod)
    assert(avgRecallProposedMethod <= 1.0 and avgRecallProposedMethod >= targetRecall)
    # assert(False)
    return avgRecallProposedMethod
    # targetRecall_fromProposedMethod = getTargetRecallFromProposedMethod(dataName, targetRecall)


# mc-checked
# assume that the missclassification cost is the same for all classes
def getTotalCostsSimple(accuracy, avgTotalFeatureCosts, misclassificationCostSymmetric, sameClassCost):
    assert(accuracy >= 0.0 and accuracy <= 1.0)
    assert(misclassificationCostSymmetric >= 0.01)
    assert(avgTotalFeatureCosts >= 0.0)
    return avgTotalFeatureCosts + (1.0 - accuracy) * misclassificationCostSymmetric + accuracy * sameClassCost



# allResults is 2D-array where each row corresponds to a different (hyper-parameter) setting and the columns are as follows
# column 0: average accuracy on validation data
# column 1: average feature costs on validation data 
# column 2: average number of false positives on validation data
# column 3: average number of false negatives on validation data
# def getBestValidTotalCostsResultAsymmetric(allResults, symmetricMisclassificationCost, sameClassCost):
#     assert(allResults.shape[1] == 2)
#     allTotalCostsValid = numpy.zeros(allResults.shape[0])
#     for i in range(allResults.shape[0]):
#         allTotalCostsValid[i] = getTotalCostsSimple(allResults[i,0], allResults[i,1], symmetricMisclassificationCost, sameClassCost)
#         
#     bestIdOnValid = numpy.argmin(allTotalCostsValid)
#      
#     validAccuracy = allResults[bestIdOnValid, 0]
#     validFeatureCosts = allResults[bestIdOnValid, 1]
#     totalCostsValidResult = getTotalCostsSimple(validAccuracy, validFeatureCosts, symmetricMisclassificationCost, sameClassCost)
#     
#     return totalCostsValidResult, validAccuracy, validFeatureCosts, bestIdOnValid





# testM = numpy.reshape(numpy.arange(10), (2,5))
# print(testM)
# print(testM.flatten())
# 
# bestValidId = 10
# GREEDY_MISER_NUMBER_OF_TREES = 10
# lambdaId = int(bestValidId / GREEDY_MISER_NUMBER_OF_TREES)
# treeId = bestValidId % GREEDY_MISER_NUMBER_OF_TREES
# print("lambdaId = ", lambdaId)
# print("treeId = ", treeId)

def getLabelsFromGreedyScores(scores):
    labels = numpy.zeros(scores.shape[0], dtype = numpy.int)
    labels[scores > 0] = 1
    return labels

def getProbabilitiesFromGreedyScores(scores):
    scoreRange = numpy.max(scores) - numpy.min(scores)
    minScore = numpy.min(scores)
    probs = (scores - minScore) / scoreRange
    return probs
    
def getProbabilitiesFromGreedyScores_avg(scores_allTrees):
    scoreRange = numpy.max(scores_allTrees) - numpy.min(scores_allTrees)
    minScore = numpy.min(scores_allTrees)
    
    scores = numpy.mean(scores_allTrees, axis = 1)
    probs = (scores - minScore) / scoreRange
    return probs

def getResultsAtTargetRecall(falsePositiveCost, targetRecall, trueLabels, allPredictedProbs, avgFeatureCosts):
    threshold_forExactRecall = getThresholdFromPredictedProbabilities(trueLabels, allPredictedProbs, targetRecall)
    predictedLabels_atExactRecall = getPredictedLabelsAtThreshold(threshold_forExactRecall, allPredictedProbs)
    
    operationCosts_exactRecall = getAverageOperationCosts(trueLabels, predictedLabels_atExactRecall, avgFeatureCosts, falsePositiveCost)
    recall_exactRecall = getRecall(trueLabels, allPredictedProbs, threshold_forExactRecall)
    fdr_exactRecall = getFDR(trueLabels, allPredictedProbs, threshold_forExactRecall)

    return operationCosts_exactRecall, fdr_exactRecall, recall_exactRecall




GREEDY_MISER_NUMBER_OF_TREES = 10
GREEDY_MISER_NUMBER_OF_LAMBDAS = 16


def getBestParametersForGreedyMiser_helper(dataName, definedFeatureCosts, testFoldId):
    
    NUMBER_OF_TRAIN_EVAL_FOLDS = 10
    
    allTrueLabels = numpy.zeros(0, dtype = numpy.int)
    
    allPredictedTrueLabelProbs = {}
    allPredictedLabel = {}
    for lambdaId in range(GREEDY_MISER_NUMBER_OF_LAMBDAS):
        for treeId in range(GREEDY_MISER_NUMBER_OF_TREES):
            allPredictedTrueLabelProbs[str(lambdaId) + "_" + str(treeId)] =  numpy.zeros(0)
            allPredictedLabel[str(lambdaId) + "_" + str(treeId)] =  numpy.zeros(0, dtype = numpy.int)
    
    allAvgFeatureCost = numpy.zeros((GREEDY_MISER_NUMBER_OF_LAMBDAS, GREEDY_MISER_NUMBER_OF_TREES))
    
    for trainFoldId in range(NUMBER_OF_TRAIN_EVAL_FOLDS):
        dataNameFull = dataName + "_" + str(trainFoldId) + "trainEvalSplitNr_" + str(testFoldId)
        allResultsInMatlab = scipy.io.loadmat(experimentSetting.MATLAB_FOLDER_RESULTS_GREEDY_MISER + dataNameFull + "_allResults")
        allAccTest_thisTrainFold = allResultsInMatlab['allAccTest']
        allAvgFeatureCost_thisTrainFold = allResultsInMatlab['allTotalCost']
        allScores_thisTrainFold = allResultsInMatlab['allScores']
        NUMBER_OF_SAMPLES = allScores_thisTrainFold.shape[1]
        assert(allScores_thisTrainFold.shape[0] == GREEDY_MISER_NUMBER_OF_LAMBDAS and allScores_thisTrainFold.shape[2] == GREEDY_MISER_NUMBER_OF_TREES)
        assert(allAvgFeatureCost_thisTrainFold.shape[0] == GREEDY_MISER_NUMBER_OF_LAMBDAS and allAvgFeatureCost_thisTrainFold.shape[1] == GREEDY_MISER_NUMBER_OF_TREES)
        assert(allAccTest_thisTrainFold.shape[0] == GREEDY_MISER_NUMBER_OF_LAMBDAS and allAccTest_thisTrainFold.shape[1] == GREEDY_MISER_NUMBER_OF_TREES)
        assert(len(allAccTest_thisTrainFold.shape) == 2 and len(allAvgFeatureCost_thisTrainFold.shape) == 2)
        
        dataInMatlab = scipy.io.loadmat(experimentSetting.MATLAB_FOLDER_DATA + dataNameFull)
        trueLabels = numpy.asarray(dataInMatlab["yte"].transpose()[0], dtype = numpy.int) # labels are in {-1, 1}
        trueLabels = experimentHelper.getLabelsStartingAtZero(trueLabels)
        
        allTrueLabels = numpy.append(allTrueLabels, trueLabels)
        
        for lambdaId in range(GREEDY_MISER_NUMBER_OF_LAMBDAS):
            for treeId in range(GREEDY_MISER_NUMBER_OF_TREES):
                scores = allScores_thisTrainFold[lambdaId, :, treeId]
                predictedLabels = getLabelsFromGreedyScores(scores)
                predictedTrueLabelProbs = getProbabilitiesFromGreedyScores(scores)
                
                allPredictedLabel[str(lambdaId) + "_" + str(treeId)] = numpy.append(allPredictedLabel[str(lambdaId) + "_" + str(treeId)], predictedLabels)
                allPredictedTrueLabelProbs[str(lambdaId) + "_" + str(treeId)] = numpy.append(allPredictedTrueLabelProbs[str(lambdaId) + "_" + str(treeId)], predictedTrueLabelProbs)
                
                allAvgFeatureCost[lambdaId, treeId] += allAvgFeatureCost_thisTrainFold[lambdaId, treeId] / float(NUMBER_OF_TRAIN_EVAL_FOLDS)
                assert(allAvgFeatureCost[lambdaId, treeId] <= numpy.sum(definedFeatureCosts)) # just to ensure that it is really the average and not a sum over all samples
                assert(allPredictedLabel[str(lambdaId) + "_" + str(treeId)].shape[0] == allPredictedTrueLabelProbs[str(lambdaId) + "_" + str(treeId)].shape[0])
                
    return  allTrueLabels, allPredictedTrueLabelProbs, allAvgFeatureCost, allPredictedLabel


def getBestParametersForGreedyMiser_targetRecall(dataName, definedFeatureCosts, testFoldId, falsePositiveCost, targetRecall, averageTrees):
    assert(not averageTrees)
    
    allTrueLabels, allPredictedTrueLabelProbs, allAvgFeatureCost, allPredictedLabel = getBestParametersForGreedyMiser_helper(dataName, definedFeatureCosts, testFoldId)
    
    operationsCostsAllFolds = numpy.zeros((GREEDY_MISER_NUMBER_OF_LAMBDAS, GREEDY_MISER_NUMBER_OF_TREES))
    
    for lambdaId in range(GREEDY_MISER_NUMBER_OF_LAMBDAS):
        for treeId in range(GREEDY_MISER_NUMBER_OF_TREES):
            assert(allTrueLabels.shape[0] == allPredictedTrueLabelProbs[str(lambdaId) + "_" + str(treeId)].shape[0])
            threshold = getThresholdFromPredictedProbabilities(allTrueLabels, allPredictedTrueLabelProbs[str(lambdaId) + "_" + str(treeId)], targetRecall)
            predictedLabels = getPredictedLabelsAtThreshold(threshold, allPredictedTrueLabelProbs[str(lambdaId) + "_" + str(treeId)])
            operationsCostsAllFolds[lambdaId, treeId] = getAverageOperationCosts(allTrueLabels, predictedLabels, allAvgFeatureCost[lambdaId, treeId], falsePositiveCost)
             
    
    bestLambdaId, bestTreeId = numpy.unravel_index( numpy.argmin(operationsCostsAllFolds), shape = operationsCostsAllFolds.shape)
    print("bestLambdaId = ", bestLambdaId)
    print("bestTreeId = ", bestTreeId)
    return bestLambdaId, bestTreeId, threshold


                
def getBestParametersForGreedyMiser_asymmetric(dataName, definedFeatureCosts, testFoldId, falsePositiveCost, falseNegativeCost):
    
    allTrueLabels, allPredictedTrueLabelProbs, allAvgFeatureCost, allPredictedLabels = getBestParametersForGreedyMiser_helper(dataName, definedFeatureCosts, testFoldId)
        
    # print("allTrueLabels = ", allTrueLabels)
    # print("allPredictedLabel = ", allPredictedLabels["0_0"])
    # assert(False)
    
    totalCostsAllFolds = numpy.zeros((GREEDY_MISER_NUMBER_OF_LAMBDAS, GREEDY_MISER_NUMBER_OF_TREES))
    
    misclassificationCosts = numpy.zeros((2, 2))
    misclassificationCosts[0, 1] = falsePositiveCost 
    misclassificationCosts[1, 0] = falseNegativeCost 
    misclassificationCosts[0, 0] = 0.0
    misclassificationCosts[1, 1] = 0.0
    
    for lambdaId in range(GREEDY_MISER_NUMBER_OF_LAMBDAS):
        for treeId in range(GREEDY_MISER_NUMBER_OF_TREES):
            assert(allTrueLabels.shape[0] == allPredictedLabels[str(lambdaId) + "_" + str(treeId)].shape[0])
            totalCostsAllFolds[lambdaId, treeId] = getAverageTotalCosts(allTrueLabels, allPredictedLabels[str(lambdaId) + "_" + str(treeId)], allAvgFeatureCost[lambdaId, treeId], misclassificationCosts)
            
    
    bestLambdaId, bestTreeId = numpy.unravel_index( numpy.argmin(totalCostsAllFolds), shape = totalCostsAllFolds.shape)
    print("bestLambdaId = ", bestLambdaId)
    print("bestTreeId = ", bestTreeId)
    return bestLambdaId, bestTreeId



# def getBestAverage10TrainFoldTotalCostsResultSymmetricForGreedyMiser(dataName, testFoldId, symmetricMisclassificationCost, sameClassCost):
#     
#     NUMBER_OF_TRAIN_EVAL_FOLDS = 10
#     
#     allResults = None
#     avgBestTotalCostsValidResult = 0.0 # used only for analysis
#     for trainFoldId in range(NUMBER_OF_TRAIN_EVAL_FOLDS):
#         dataNameFull = dataName + "_" + str(trainFoldId) + "trainEvalSplitNr_" + str(testFoldId) + "_allResults"
#         allResultsInMatlab = scipy.io.loadmat(experimentSetting.MATLAB_DATA_FOLDER_RESULTS + "greedyMiser/" + dataNameFull)
#         allAccTest = allResultsInMatlab['allAccTest']
#         allTotalCost = allResultsInMatlab['allTotalCost']
#         allScores = allResultsInMatlab['allScores']
#         
#         assert(allAccTest.shape[0] == GREEDY_MISER_NUMBER_OF_LAMBDAS) # number of lambda parameters
#         assert(allAccTest.shape[1] == GREEDY_MISER_NUMBER_OF_TREES) # number of trees
#         totalNumberOfClassifiers = (allAccTest.flatten()).shape[0]
#         allResultsOneTrainFold = numpy.zeros((totalNumberOfClassifiers, 2))
#         allResultsOneTrainFold[:,0] = allAccTest.flatten()
#         allResultsOneTrainFold[:,1] = allTotalCost.flatten()
#     
#         if allResults is None:
#             allResults = numpy.zeros_like(allResultsOneTrainFold)
#         allResults += allResultsOneTrainFold
#         
#         bestTotalCostsValidResult, _, _, _ = getBestValidTotalCostsResultSymmetric(allResultsOneTrainFold, symmetricMisclassificationCost, sameClassCost)
#         avgBestTotalCostsValidResult += bestTotalCostsValidResult
#     
#     allResults = allResults / float(NUMBER_OF_TRAIN_EVAL_FOLDS)
#     
#     avgBestTotalCostsValidResult = avgBestTotalCostsValidResult / float(NUMBER_OF_TRAIN_EVAL_FOLDS)
#     print("avgBestTotalCostsValidResult = ", avgBestTotalCostsValidResult)
#     # assert(False)
#     
#     totalCostsValidResult, validAccuracy, validFeatureCosts, bestValidId = getBestValidTotalCostsResultSymmetric(allResults, symmetricMisclassificationCost, sameClassCost)
#     
#     bestLambdaId = int(bestValidId / GREEDY_MISER_NUMBER_OF_TREES)
#     bestTreeId = bestValidId % GREEDY_MISER_NUMBER_OF_TREES
#     print("best bestLambdaId = ", bestLambdaId)
#     print("best treeId = ", bestTreeId)
# 
#     return totalCostsValidResult, validAccuracy, validFeatureCosts, bestLambdaId, bestTreeId, avgBestTotalCostsValidResult


def getAverageTotalCosts_forADAPTGBRT(avgFeatureCost, avgFalsePositives, avgFalseNegatives, definedFalsePositiveCost, definedFalseNegativeCost):
    return avgFeatureCost + (avgFalsePositives * definedFalsePositiveCost +  avgFalseNegatives * definedFalseNegativeCost)
    


def getBestAverage10TrainFoldTotalCostsResultAsymmetricForADAPTGBRT(classificationModelName, dataName, testFoldId, falsePositiveCost, falseNegativeCost):
    
    allResults = None
    for trainFoldId in range(constants.NUMBER_OF_TRAIN_EVAL_FOLDS_MATLAB_METHODS):
        dataNameFull = dataName + "_" + str(trainFoldId) + "trainEvalSplitNr_" + str(testFoldId) + "_" + classificationModelName
        allResultsOneTrainFold = numpy.loadtxt(experimentSetting.MATLAB_FOLDER_RESULTS_ADAPT_GBRT + dataNameFull + "_" + "allResults.csv", delimiter = ",")
        if allResults is None:
            allResults = numpy.zeros_like(allResultsOneTrainFold)
        allResults += allResultsOneTrainFold
     
    
    allResults = allResults / float(constants.NUMBER_OF_TRAIN_EVAL_FOLDS_MATLAB_METHODS)
    
    print("allResults = ", allResults)
    print("best accuracy result = ", allResults[numpy.argmax(allResults[0,:])])
    
    assert(allResults.shape[0] > 10 and allResults.shape[1] == 4)
    allTotalCosts = numpy.zeros(allResults.shape[0])
    for paramSettingId in range(allResults.shape[0]):
        allTotalCosts[paramSettingId] = getAverageTotalCosts_forADAPTGBRT(allResults[paramSettingId, 1], allResults[paramSettingId, 2], allResults[paramSettingId, 3], falsePositiveCost, falseNegativeCost)
    
    bestIdOnValid = numpy.argmin(allTotalCosts)
    return bestIdOnValid


# reading checked
def getBestAverage10TrainFoldTotalCostsResultSymmetricForADAPTGBRT(classificationModelName, dataName, testFoldId, symmetricMisclassificationCost, sameClassCost):
     
    
    allResults = None
    for trainFoldId in range(constants.NUMBER_OF_TRAIN_EVAL_FOLDS_MATLAB_METHODS): 
        dataNameFull = dataName + "_" + str(trainFoldId) + "trainEvalSplitNr_" + str(testFoldId) + "_" + classificationModelName
        allResultsOneTrainFold = numpy.loadtxt(experimentSetting.MATLAB_FOLDER_RESULTS_ADAPT_GBRT + dataNameFull + "_" + "allResults.csv", delimiter = ",")
        if allResults is None:
            allResults = numpy.zeros_like(allResultsOneTrainFold)
        allResults += allResultsOneTrainFold
    
     
    allResults = allResults / float(constants.NUMBER_OF_TRAIN_EVAL_FOLDS_MATLAB_METHODS)
    
    print("allResults = ", allResults)
    print("best accuracy result = ", allResults[numpy.argmax(allResults[0,:])])
    
    bestIdOnValid = getBestValidTotalCostsResultSymmetric(allResults, symmetricMisclassificationCost, sameClassCost)
    return bestIdOnValid


# mc-checked
# allResults is 2D-array where each row corresponds to a different (hyper-parameter) setting and the columns are as follows
# column 0: average accuracy on validation data
# column 1: average feature costs on validation data 
# column 2: average number of false positives on validation data
# column 3: average number of false negatives on validation data
def getBestValidTotalCostsResultSymmetric(allResults, symmetricMisclassificationCost, sameClassCost):
    assert(allResults.shape[0] > 10 and allResults.shape[1] == 4)
    
    allTotalCostsValid = numpy.zeros(allResults.shape[0])
    for i in range(allResults.shape[0]):
        allTotalCostsValid[i] = getTotalCostsSimple(allResults[i,0], allResults[i,1], symmetricMisclassificationCost, sameClassCost)
         
    bestIdOnValid = numpy.argmin(allTotalCostsValid)
      
    # validAccuracy = allResults[bestIdOnValid, 0]
    # validFeatureCosts = allResults[bestIdOnValid, 1]
    # totalCostsValidResult = getTotalCostsSimple(validAccuracy, validFeatureCosts, symmetricMisclassificationCost, sameClassCost)
     
    return bestIdOnValid


# mc-checked
# allResults is 2D-array where each row corresponds to a different (hyper-parameter) setting and the columns are as follows
# column 0: average accuracy on validation data
# column 1: average feature costs on validation data 
# column 2: average accuracy on test data
# column 3: average feature costs on test data 
# def getBestTotalCostsResultSymmetric(allResults, symmetricMisclassificationCost, sameClassCost):
#     assert(allResults.shape[1] == 4)
#     allTotalCostsValid = numpy.zeros(allResults.shape[0])
#     for i in range(allResults.shape[0]):
#         allTotalCostsValid[i] = getTotalCostsSimple(allResults[i,0], allResults[i,1], symmetricMisclassificationCost, sameClassCost)
#          
#     bestIdOnValid = numpy.argmin(allTotalCostsValid)
#       
#     testAccuracy = allResults[bestIdOnValid, 2]
#     testFeatureCosts = allResults[bestIdOnValid, 3]
#     totalCostsTestResult = getTotalCostsSimple(testAccuracy, testFeatureCosts, symmetricMisclassificationCost, sameClassCost)
#       
#     totalCostsValidResult = getTotalCostsSimple(allResults[bestIdOnValid, 0], allResults[bestIdOnValid, 1], symmetricMisclassificationCost, sameClassCost)
#       
#     return totalCostsTestResult, testAccuracy, testFeatureCosts, bestIdOnValid, totalCostsValidResult


def saveBestHyperparameterStringForADAPTGBRT(bestValidId, classificationModelName, dataName, foldId, falsePositiveCost, COST_TYPE):
    filename = experimentSetting.PARAMETER_FOLDER_ADAPT_GBRT + "standardParamRange"
    with open(filename, "r") as f:
        for lineId, line in enumerate(f):
            if lineId == bestValidId:
                bestHyperparametersStr = line.strip()
                break
     
    filename = experimentSetting.PARAMETER_FOLDER_ADAPT_GBRT + "standardParamRange_" + dataName + "_" + str(int(falsePositiveCost)) + '_forFinalTrainingAndTesting_' + str(foldId) + "_" + COST_TYPE + "_" + classificationModelName
     
    print("bestHyperparametersStr = ", bestHyperparametersStr)
    
    with open(filename, "w") as f:
        f.write(bestHyperparametersStr)
     
    print("saved best configuration to ", filename)
    return



# checked for applicability to target recall
def getAverageMisclassificationCosts(trueLabels, predictedLabels, misclassificationCosts):
    assert(numpy.max(trueLabels) == misclassificationCosts.shape[0] - 1)
    n = trueLabels.shape[0]
    assert(n > 1 and predictedLabels.shape[0] == n)
    
    totalMisclassificationCosts = 0.0
    for i in range(n):
        totalMisclassificationCosts += misclassificationCosts[trueLabels[i], predictedLabels[i]]
     
    return (totalMisclassificationCosts / float(n))

# checked for applicability to target recall
def getAverageTotalCosts(trueLabels, predictedLabels, averageFeatureCosts, misclassificationCosts):
    return averageFeatureCosts + getAverageMisclassificationCosts(trueLabels, predictedLabels, misclassificationCosts) 


def getAverageOperationCosts(trueLabels, predictedLabels, averageFeatureCosts, falsePositiveCost):
    n = trueLabels.shape[0]
    assert(n > 1 and predictedLabels.shape[0] == n)
    
    totalFPCosts = 0.0
    for i in range(n):
        assert(trueLabels[i] == 0 or trueLabels[i] == 1)
        assert(predictedLabels[i] == 0 or predictedLabels[i] == 1)
        if predictedLabels[i] == 1 and trueLabels[i] == 0:
            totalFPCosts += falsePositiveCost
     
    averageFalsePositiveCosts = (totalFPCosts / float(n)) 
    return averageFeatureCosts + averageFalsePositiveCosts
    

# mc-checked
def getBestTotalCostsResultGeneral(evalLabels, allEvalPredictedLabels, allEvalFeatureCosts, testLabels, allTestPredictedLabels, allTestFeatureCosts, misclassificationCosts):
    numberOfSettings = len(allEvalPredictedLabels)
    allTotalCostsValid = numpy.zeros(numberOfSettings)
    for i in range(numberOfSettings):
        allTotalCostsValid[i] = getAverageTotalCosts(evalLabels, allEvalPredictedLabels[i], allEvalFeatureCosts[i], misclassificationCosts)
    bestIdOnEval = numpy.argmin(allTotalCostsValid)
    
    totalCostsTestResult = getAverageTotalCosts(testLabels, allTestPredictedLabels[bestIdOnEval], allTestFeatureCosts[bestIdOnEval], misclassificationCosts)
    
    return totalCostsTestResult, bestIdOnEval



def showAsStr(floatNumber):
    return str(round(floatNumber, 2))

def showAsStrDetailed(floatNumber, digits):
    return str(round(floatNumber, digits))

def showHelper(info, allResults):
    print(info + showAsStr(numpy.average(allResults)) + " (" + showAsStr(numpy.std(allResults)) + ")")
    return

def getDetailedStr(allResults):
    digits = 4
    return showAsStrDetailed(numpy.average(allResults), digits) + " (" + showAsStrDetailed(numpy.std(allResults), digits) + ")"
    
def showHelperDetailed(info, allResults):
    digits = 4
    print(info + showAsStrDetailed(numpy.average(allResults), digits) + " (" + showAsStrDetailed(numpy.std(allResults), digits) + ")")
    return

def showResults(allMissclassificationCosts, allAccuracies):
    showHelper("misclassification costs = ", allMissclassificationCosts)
    showHelper("accuracy = ", allAccuracies)
    return

def getAverageHeldOutLogLikelihood(classificationModel, evalData, evalLabels):
    evalDataProbs = classificationModel.predict_proba(evalData)
    return getAverageHeldOutLogLikelihood_FromProbabilities(evalDataProbs, evalLabels)
       

def getAverageHeldOutLogLikelihood_FromProbabilities(evalDataProbs, evalLabels):
    assert(evalDataProbs.shape[0] == evalLabels.shape[0] and evalDataProbs.shape[1] == 2)
    correctClassProbs = evalDataProbs[numpy.arange(evalDataProbs.shape[0]), evalLabels]
    if numpy.all(correctClassProbs > 0):
        return numpy.average(numpy.log(correctClassProbs))
    else:
        return float("-inf")


def getAverageHeldOutLogLikelihood_FromTrueProbabilities(predictedProbTrueLabel, evalLabels):
    assert(predictedProbTrueLabel.shape[0] == evalLabels.shape[0] and len(predictedProbTrueLabel.shape) == 1)
    predictedProbs = numpy.vstack((predictedProbTrueLabel, predictedProbTrueLabel)).transpose()
    predictedProbs[:,0] = 1.0 - predictedProbs[:,0]
    return getAverageHeldOutLogLikelihood_FromProbabilities(predictedProbs, evalLabels)

    
def eval_NN(model, data, trueLabels):
    predictedProbTrueLabel = model.predict(data)[:,0]
    predictedProbs = numpy.vstack((predictedProbTrueLabel, predictedProbTrueLabel)).transpose()
    predictedProbs[:,0] = 1.0 - predictedProbs[:,0]
    
    auc = sklearn.metrics.roc_auc_score(trueLabels, predictedProbTrueLabel)
    logLikelihood = getAverageHeldOutLogLikelihood_FromProbabilities(predictedProbs, trueLabels)
     
    return auc, logLikelihood
        


# re-checked
# mc-checked
# def getTestDataAverageTotalCostsWithFixedFeatureIds(classificationModel, evalDataFullCovariates, evalLabels, misclassificationCosts, selectedFeatureIds, definedFeatureCosts):
#     evalData = evalDataFullCovariates[:, selectedFeatureIds]
#     averageMisclassificationCosts = getTestDataPerformance(classificationModel, evalData, evalLabels, misclassificationCosts)
#     avgTotalFeatureCosts = numpy.sum(definedFeatureCosts[selectedFeatureIds])
#     return averageMisclassificationCosts + avgTotalFeatureCosts, avgTotalFeatureCosts, averageMisclassificationCosts

# re-checked
# mc-checked
# def getTestDataPerformance(classificationModel, evalData, evalLabels, misclassificationCosts):
#     predictedLabels = classificationModel.predict(evalData)
#     averageMisclassificationCosts = getAverageMisclassificationCosts(evalLabels, predictedLabels, misclassificationCosts)
#     # averageHeldOutLL = getAverageHeldOutLogLikelihood(classificationModel, evalData, evalLabels)
#     
#     # print("Test Data Results: ")
#     # print("averageMisclassificationCosts = ", averageMisclassificationCosts)
#     # print("averageAccuracy = ", averageAccuracy)
#     # print("averageHeldOutLL = ", averageHeldOutLL)
#     return averageMisclassificationCosts  


    

# checked for applicability to target recall
# mc-checked
def bayesClassifier(allPredictedClassProbabilities, misclassificationCosts):
    allPredictedClassProbabilities = ensure2D(allPredictedClassProbabilities)
    n = allPredictedClassProbabilities.shape[0]
    c = allPredictedClassProbabilities.shape[1]
    assert(misclassificationCosts.shape[0] == c and misclassificationCosts.shape[1] == c)
    
    predictedLabels = numpy.zeros(n)
    for i in range(n):
        predictedClassProbabilities = allPredictedClassProbabilities[i]
        
        bayesRisks = numpy.zeros(c)
        for predictedLabel in range(c):
            bayesRisks[predictedLabel] = numpy.sum(misclassificationCosts[:, predictedLabel] * predictedClassProbabilities)
        
        predictedLabels[i] = numpy.argmin(bayesRisks)
        
    return predictedLabels.astype(numpy.int)


# checked for applicability to target recall
# mc-checked
def bayesRisk(allPredictedClassProbabilities, misclassificationCosts):
    n = allPredictedClassProbabilities.shape[0]
    assert(allPredictedClassProbabilities.shape[1] == misclassificationCosts.shape[0])
    
    allPredictedLabels = bayesClassifier(allPredictedClassProbabilities, misclassificationCosts)
    assert(allPredictedLabels.shape[0] == n)
    
    bayesRiskEstimate = 0.0
    for i in range(n):
        predictedLabel = allPredictedLabels[i]
        predictedClassProbabilities = allPredictedClassProbabilities[i]
        bayesRiskEstimate += numpy.sum(misclassificationCosts[:, predictedLabel] * predictedClassProbabilities)  
    
    bayesRiskEstimate = bayesRiskEstimate / float(n)
    return bayesRiskEstimate



# READING CHECKED
def thresholdClassifier(allPredictedClassProbabilities, threshold):
    allPredictedClassProbabilities = ensure2D(allPredictedClassProbabilities)
    n = allPredictedClassProbabilities.shape[0]
    assert(allPredictedClassProbabilities.shape[1] == 2)
    
    predictedLabels = numpy.zeros(n)
    predictedLabels[allPredictedClassProbabilities[:,1] >= threshold] = 1
        
    return predictedLabels.astype(numpy.int)


# READING CHECKED
def expectedFalsePositiveCosts_underThresholdRequirement(allPredictedClassProbabilities, threshold, falsePositiveCost):
    n = allPredictedClassProbabilities.shape[0]
    assert(allPredictedClassProbabilities.shape[1] == 2)
    
    allPredictedLabels = thresholdClassifier(allPredictedClassProbabilities, threshold)
    assert(allPredictedLabels.shape[0] == n)
    
    expectedFalsePositiveCosts = 0.0
    for i in range(n):
        if allPredictedLabels[i] == 1:
            predictedNegativeClassProbability = allPredictedClassProbabilities[i, 0]
            expectedFalsePositiveCosts += predictedNegativeClassProbability * falsePositiveCost
    
    expectedFalsePositiveCosts = expectedFalsePositiveCosts / float(n)
    return expectedFalsePositiveCosts



# checked for applicability to target recall
def getOverallPerformance_fixedCovariateSet(classificationModel, evalDataFullCovariates, evalLabels, definedFeatureCosts, misclassificationCosts, selectedFeatureIds, targetRecall):
    evalData = evalDataFullCovariates[:, selectedFeatureIds]
    predictedProbs = classificationModel.predict_proba(evalData)
    predictedLabels = bayesClassifier(predictedProbs, misclassificationCosts)

    averageMisclassificationCosts = getAverageMisclassificationCosts(evalLabels, predictedLabels, misclassificationCosts)
    avgFeatureCosts = numpy.sum(definedFeatureCosts[selectedFeatureIds])
    
    predictedProbTrueLabel = predictedProbs[:,1]
    auc = sklearn.metrics.roc_auc_score(evalLabels, predictedProbTrueLabel)
    accuracy = sklearn.metrics.accuracy_score(evalLabels, predictedLabels)
    recall = sklearn.metrics.recall_score(evalLabels, predictedLabels)
    FDR = 1.0 - sklearn.metrics.precision_score(evalLabels, predictedLabels)
    operationCosts = getAverageOperationCosts(evalLabels, predictedLabels, avgFeatureCosts, misclassificationCosts[0,1])
      
    threshold_forExactRecall = getThresholdFromPredictedProbabilities(evalLabels, predictedProbTrueLabel, targetRecall)
    predictedEvalLabels_atExactRecall = getPredictedLabelsAtThreshold(threshold_forExactRecall, predictedProbTrueLabel)
    testRecallAllFolds_exactRecall = getRecall(evalLabels, predictedProbTrueLabel, threshold_forExactRecall)
    testFDRAllFolds_exactRecall = getFDR(evalLabels, predictedProbTrueLabel, threshold_forExactRecall)
    testOperationCostsAllFolds_exactRecall = getAverageOperationCosts(evalLabels, predictedEvalLabels_atExactRecall, avgFeatureCosts, misclassificationCosts[0,1])
  
    return averageMisclassificationCosts + avgFeatureCosts, avgFeatureCosts, averageMisclassificationCosts, accuracy, auc, recall, FDR, operationCosts, testRecallAllFolds_exactRecall, testFDRAllFolds_exactRecall, testOperationCostsAllFolds_exactRecall



#         if classificationModelName == "logReg":
#             bestModel, bestAlphaValue, _ = evaluation.getBestL2RegularizedLogisticRegressionModelNew(trainData, trainLabels)
#             threshold = evaluation.getThresholdEstimate(trainData, trainLabels, bestAlphaValue, [targetRecall], modelType = classificationModelName)[0]
#         elif classificationModelName == "GAM":
#             bestModel = evaluation.getBestGAM(trainData, trainLabels)
#             threshold = evaluation.getThresholdEstimate(trainData, trainLabels, bestModel, [targetRecall], modelType = classificationModelName)[0]


def nonLinearFeatureSelection_withGAM(data, labels, definedFeatureCosts):
    model, _ = getBestGAM_withoutCV(data, labels, selectedFeatureList = None)
    
    NR_COVARIATES = data.shape[1]
    NR_SPLINES = 20
    assert(numpy.all(numpy.asarray(model.n_splines) == NR_SPLINES))
    assert(len(model.terms._terms) == data.shape[1] + 1)
    assert(model.coef_.shape[0] == NR_SPLINES * data.shape[1] + 1)
    
    lastCoeffId = NR_SPLINES * data.shape[1]
    
    splineBasis = model.terms.build_columns(data)
    transformedData = splineBasis.todense()[:, 0:lastCoeffId] # remove last id which corresponds to bias
    transformedData = numpy.asarray(transformedData)
    
    
    allMeans = numpy.nanmean(transformedData, axis = 0)
    allStds = numpy.nanstd(transformedData, axis = 0)
    allStds[allStds == 0] = 1.0
    assert(not numpy.any(numpy.isnan(allMeans)))
    assert(not numpy.any(numpy.isnan(allStds)))
    assert(numpy.all(allStds) > 0.0)
    transformedData = preprocessing.standardizeDataWithMeanStd(allMeans, allStds, transformedData)
    
    # scale with inverse feature costs
    definedFeatureCostsWithGroups = numpy.repeat(definedFeatureCosts, NR_SPLINES)
    transformedData = transformedData / definedFeatureCostsWithGroups
    
    # print("definedFeatureCostsWithGroups.shape = ", definedFeatureCostsWithGroups.shape[0])
    # print("transformedData.shape = ", transformedData.shape[1])
    # print("definedFeatureCosts = ", definedFeatureCosts)
    # print("definedFeatureCostsWithGroups = ", definedFeatureCostsWithGroups)
    # assert(False)
    
    print("shape = ", transformedData.shape)
    print("lastCoeffId = ", lastCoeffId)
    assert(transformedData.shape[0] == data.shape[0] and transformedData.shape[1] == NR_SPLINES * data.shape[1])
    
    groups = numpy.array([[group]*NR_SPLINES for group in range(NR_COVARIATES)]).ravel()
    # print("groups = ")
    # print(groups)
    
    labels = numpy.reshape(labels, (-1, 1))
    
    import group_lasso
    
    allFeatureSetsInOrder = []
    
    allLambdaValues = numpy.geomspace(start = 0.0001, stop = 0.02, num=100)
    allLambdaValues = numpy.flip(allLambdaValues)
    previousCoeff = numpy.zeros(NR_COVARIATES * NR_SPLINES)
    for lambdaValue in allLambdaValues:
        gl = group_lasso.LogisticGroupLasso(groups, group_reg=lambdaValue, l1_reg=0.001, n_iter=100, tol=1e-05, subsampling_scheme=None, fit_intercept=True, random_state=12352934, warm_start=False)
        gl.fit(transformedData, labels)
        selectedDims = gl.sparsity_mask
        
        selectedGroups = numpy.ones(NR_COVARIATES)
        for i in range(NR_COVARIATES):
            startId = i * NR_SPLINES
            endId = (i+1) * NR_SPLINES
            selectedGroups[i] = numpy.any(selectedDims[startId:endId])
        
        print("selectedGroups = ", selectedGroups)
    
        selectedVariableIds = numpy.where(selectedGroups != 0)[0]
        print("selectedVariableIds = ", selectedVariableIds)
        # assert(False)
        
        notAlreadyFound = True
        for alreadyFoundFeatureSet in allFeatureSetsInOrder:
            if set(selectedVariableIds) == set(alreadyFoundFeatureSet):
                notAlreadyFound = False
                break
            
        if notAlreadyFound:
            allFeatureSetsInOrder.append(selectedVariableIds)
    
    # check if [] and [0,1,2,...p-1] is contained
    containsEmptySet = False
    containsFullSet = False
    allFeatures = numpy.arange(NR_COVARIATES)
    for selectedVariableIds in allFeatureSetsInOrder:
        if set(selectedVariableIds) == set(allFeatures):
            containsFullSet = True
        elif set(selectedVariableIds) == set([]):
            containsEmptySet = True
    
    if not containsEmptySet:
        allFeatureSetsInOrder.append([])
    if not containsFullSet:
        allFeatureSetsInOrder.append(allFeatures)
    
    allFeatureSetsInOrder = sorted(allFeatureSetsInOrder, key=lambda x: x.shape[0] )
    
    return allFeatureSetsInOrder

     
    # print("selectedGroups.shape[0] = ", selectedGroups.shape[0])
    # print("NR_COVARIATES = ", NR_COVARIATES)
    # assert(selectedGroups.shape[0] == NR_COVARIATES)
    # print("gl.sparsity_mask size = ", gl.sparsity_mask.shape[0])
    # print("coefficients = ", gl.coef_)
    
#     queryData_splineBasis = splineBasis[:, startId:endId]
#     
#     densityTrainDataResponse = numpy.matmul(queryData_splineBasis, queryBetaPart)
#     assert(densityTrainDataResponse.shape[0] == 1 and densityTrainDataResponse.shape[1] == allData.shape[0])
#     densityTrainDataResponse = numpy.asarray(densityTrainDataResponse)[0]
#     
#     covMatrix = numpy.cov(queryData_splineBasis.transpose())
    

                
def getBestGAM_withoutCV(data, labels, selectedFeatureList = None):
    
    if selectedFeatureList is not None:
        data = data[:, selectedFeatureList]
    
    pyGamValues = 10
    trainedModel = LogisticGAM()
    
    scores = trainedModel.gridsearch(data, labels, lam = numpy.logspace(-3, 5, pyGamValues), objective='AICc', keep_best=True, return_scores=True)
    # print("scores = ", scores)
    # assert(False)
    
    bestScore = float("inf")
    bestModel = None
    for model in scores:
        print("*****************************")
        print("model = ", model)
        print("score = ", scores[model])
        if scores[model] < bestScore:
            bestScore = scores[model]
            bestModel = model
    
    return bestModel, bestScore
    

def getModel_helper(trainData, trainLabels, usePyGAM, lambdaValue):
    
    if usePyGAM:
        trainedModel = LogisticGAM()
        trainedModel.set_params(lam = lambdaValue, force = True)
        
        try:
            trainedModel.fit(trainData, trainLabels)
        except ValueError as error:
            print("VALUE ERROR - NO BEST MODEL FOUND IN PYGAM - use a basic model")
            trainedModel = sklearn.linear_model.LogisticRegression(penalty="l2",C= 1.0)
            trainedModel.fit(trainData, trainLabels)
            return trainedModel
        except AttributeError as error:
            print("AttributeError - NO BEST MODEL FOUND IN PYGAM - use a basic model")
            trainedModel = sklearn.linear_model.LogisticRegression(penalty="l2",C= 1.0)
            trainedModel.fit(trainData, trainLabels)
            return trainedModel
        except numpy.linalg.LinAlgError as error:
            print("LinAlgError - NO BEST MODEL FOUND IN PYGAM - use a basic model")
            trainedModel = sklearn.linear_model.LogisticRegression(penalty="l2",C= 1.0)
            trainedModel.fit(trainData, trainLabels)
            return trainedModel
        
    else:
        # run logistic regression
        alphaValue = 2 ** lambdaValue
        trainedModel = sklearn.linear_model.LogisticRegression(penalty="l2",C= 1.0 / alphaValue)
        trainedModel.fit(trainData, trainLabels)
        
    return trainedModel



# PREVIOUS VERSION
# def getModel_helper(trainData, trainLabels, logAlphaValue, pyGamValues):
#     if logAlphaValue == -numpy.inf:
#         
#         trainedModel = LogisticGAM()
#                 
#         try:
#             trainedModel.gridsearch(trainData, trainLabels, lam = numpy.logspace(-3, 5, pyGamValues), objective='UBRE', keep_best=True)
#         except ValueError as error:
#             print("VALUE ERROR - NO BEST MODEL FOUND IN PYGAM - use a basic model")
#             trainedModel = sklearn.linear_model.LogisticRegression(penalty="l2",C= 1.0)
#             trainedModel.fit(trainData, trainLabels)
#             return trainedModel
#         except AttributeError as error:
#             print("AttributeError - NO BEST MODEL FOUND IN PYGAM - use a basic model")
#             trainedModel = sklearn.linear_model.LogisticRegression(penalty="l2",C= 1.0)
#             trainedModel.fit(trainData, trainLabels)
#             return trainedModel
#         except numpy.linalg.LinAlgError as error:
#             print("LinAlgError - NO BEST MODEL FOUND IN PYGAM - use a basic model")
#             trainedModel = sklearn.linear_model.LogisticRegression(penalty="l2",C= 1.0)
#             trainedModel.fit(trainData, trainLabels)
#             return trainedModel
#     else:
#         # run logistic regression
#         alphaValue = 2 ** logAlphaValue
#         trainedModel = sklearn.linear_model.LogisticRegression(penalty="l2",C= 1.0 / alphaValue)
#         trainedModel.fit(trainData, trainLabels)
#         
#     return trainedModel



    

# mc-checked
def getBestModel(data, labels, classificationModelName):
    assert(numpy.all(numpy.logical_or(labels == 0, labels == 1)))
    assert(classificationModelName == "logReg" or classificationModelName == "GAM" or classificationModelName == "Combined")

    misclassificationCosts = None

    PYGAM_NR_LAMBDA_VALUES = 10

    NUMBER_OF_FOLDS = 10
    kFoldMaster = sklearn.model_selection.StratifiedKFold(n_splits = NUMBER_OF_FOLDS,  shuffle=True, random_state=4324232)
        
    lambdaValuesPYGAM = numpy.logspace(-3, 5, PYGAM_NR_LAMBDA_VALUES)
    lambdaValuesLogReg = numpy.arange(-10.0, 10.0, step = 0.5)

    if classificationModelName == "Combined":
        lambdaValues = numpy.hstack((lambdaValuesPYGAM, lambdaValuesLogReg))
        pyGamIndicators = numpy.asarray([False] * lambdaValues.shape[0]) 
        pyGamIndicators[0:lambdaValuesPYGAM.shape[0]] = True
    elif classificationModelName == "GAM":
        lambdaValues = lambdaValuesPYGAM
        pyGamIndicators = numpy.asarray([True] * lambdaValues.shape[0])
    elif classificationModelName == "logReg":
        lambdaValues = lambdaValuesLogReg
        pyGamIndicators = numpy.asarray([False] * lambdaValues.shape[0])
    else:
        assert(False)
  
  
    # flip: ensures that we prefer higher regularization in case where CV performance is the same.
    lambdaValues = numpy.flip(lambdaValues, axis = 0)  
    pyGamIndicators = numpy.flip(pyGamIndicators, axis = 0)
    
    allResults = numpy.zeros(lambdaValues.shape[0])
    allPredictedProbs = -1.0 * numpy.ones((lambdaValues.shape[0], labels.shape[0]))
  
    for modelId in range(lambdaValues.shape[0]):
        
        allLogOutLL = numpy.zeros(NUMBER_OF_FOLDS)
        allAvgMisclassificationCosts = numpy.zeros(NUMBER_OF_FOLDS)
        
        for foldId, (train_index, eval_index) in enumerate(kFoldMaster.split(data, labels)):
            trainData = data[train_index]
            trainLabels = labels[train_index]
            evalData = data[eval_index]
            evalLabels = labels[eval_index]
            
            trainedModel = getModel_helper(trainData, trainLabels, pyGamIndicators[modelId], lambdaValues[modelId])
           
            evalDataProbsTrueLabel = getPredictedProb(trainedModel, evalData)
            
            allPredictedProbs[modelId, eval_index] = evalDataProbsTrueLabel
            
            # allLogOutLL[foldId] = sklearn.metrics.roc_auc_score(evalLabels, evalDataProbsTrueLabel) 
            allLogOutLL[foldId] = getAverageHeldOutLogLikelihood_FromTrueProbabilities(evalDataProbsTrueLabel, evalLabels) 
            
            if misclassificationCosts is not None:
                predictedLabels = bayesClassifier(evalDataProbsTrueLabel, misclassificationCosts)
                allAvgMisclassificationCosts[foldId] = getAverageMisclassificationCosts(evalLabels, predictedLabels, misclassificationCosts) 
        
        if misclassificationCosts is not None:
            allResults[modelId] = - numpy.average(allAvgMisclassificationCosts)
        else:
            allResults[modelId] = numpy.average(allLogOutLL)
        
    
    bestId = numpy.argmax(allResults)
    assert(allResults[bestId] > float("-inf") and allResults[bestId] < float("inf"))

    finalModel = getModel_helper(data, labels, pyGamIndicators[bestId], lambdaValues[bestId])
    
    predictedProbsBestModel = allPredictedProbs[bestId,:]
    assert(numpy.all(predictedProbsBestModel != -1))
    
    return finalModel, predictedProbsBestModel



# PREVIOUS VERSION
# def getBestModel(data, labels, pyGamValues, classificationModelName):
#     assert(classificationModelName == "logReg" or classificationModelName == "GAM" or classificationModelName == "Combined")
# 
#     NUMBER_OF_FOLDS = 10
#     kFoldMaster = sklearn.model_selection.StratifiedKFold(n_splits = NUMBER_OF_FOLDS,  shuffle=True, random_state=4324232)
#     
#     if classificationModelName == "Combined":
#         lambdaRegValues = numpy.arange(-10.0, 10.0, step = 0.5)
#         lambdaRegValues = numpy.flip(lambdaRegValues, axis = 0)
#         lambdaRegValues = numpy.append(lambdaRegValues, [-numpy.inf])  # -numpy.inf is used to indicate GAM model  
#     elif classificationModelName == "logReg":
#         lambdaRegValues = numpy.arange(-10.0, 10.0, step = 0.5)
#         lambdaRegValues = numpy.flip(lambdaRegValues, axis = 0)
#     elif classificationModelName == "GAM":
#         lambdaRegValues = numpy.asarray([-numpy.inf])  # -numpy.inf is used to indicate GAM model  
#     else:
#         assert(False)
#         
#     
#     allResults = numpy.zeros(lambdaRegValues.shape[0])
#     allPredictedProbs = -1.0 * numpy.ones((lambdaRegValues.shape[0], labels.shape[0]))
#   
#     allResultsRecorded = []
#     for modelId, logAlphaValue in enumerate(lambdaRegValues):
#         
#         allLogOutLL = numpy.zeros(NUMBER_OF_FOLDS)
#         # allAccuracies = numpy.zeros(NUMBER_OF_FOLDS)
#         
#         for foldId, (train_index, eval_index) in enumerate(kFoldMaster.split(data, labels)):
#             trainData = data[train_index]
#             trainLabels = labels[train_index]
#             evalData = data[eval_index]
#             evalLabels = labels[eval_index]
#             
#             trainedModel = getModel_helper(trainData, trainLabels, logAlphaValue, pyGamValues)
#            
#             evalDataProbsTrueLabel = getPredictedProb(trainedModel, evalData)
#             
#             allPredictedProbs[modelId, eval_index] = evalDataProbsTrueLabel 
#             allLogOutLL[foldId] = getAverageHeldOutLogLikelihood_FromTrueProbabilities(evalDataProbsTrueLabel, evalLabels) 
#             # allAccuracies[foldId] = getTestAccuracy(trainedModel, evalData, evalLabels)
#         
#         
#         allResults[modelId] = numpy.average(allLogOutLL)
#         
#         # allResultsRecorded.append(allAccuracies)
#         # allResults[modelId] = numpy.average(allAccuracies)
#         
#     bestId = numpy.argmax(allResults)
#     assert(allResults[bestId] > float("-inf") and allResults[bestId] < float("inf"))
# 
#     # for modelId, logAlphaValue in enumerate(lambdaRegValues):
#     #     results = allResultsRecorded[modelId]
#     #     print("logAlphaValue =  " + str(logAlphaValue) + ", results = " + str(results) + ", average = " + str(numpy.average(results)))
#     # assert(False)
#     
#     finalModel = getModel_helper(data, labels, lambdaRegValues[bestId], pyGamValues)
#     
#     predictedProbsBestModel = allPredictedProbs[bestId,:]
#     assert(numpy.all(predictedProbsBestModel != -1))
#     
#     return finalModel, predictedProbsBestModel



# mc-checked
def getBestL2RegularizedLogisticRegressionModelNew(data, labels, misclassificationCosts = None):
    
    evalUsingLogLL = True # MUST BE SET TO TRUE !!
    NUMBER_OF_FOLDS = 10
    kFoldMaster = sklearn.model_selection.StratifiedKFold(n_splits = NUMBER_OF_FOLDS,  shuffle=True, random_state=4324232)
    
    lambdaRegValues = numpy.arange(-10.0, 10.0, step = 0.5)
    lambdaRegValues = numpy.flip(lambdaRegValues, axis = 0)
    
    allResults = numpy.zeros((lambdaRegValues.shape[0], 3))
    
    for regId, logAlphaValue in enumerate(lambdaRegValues):
        alphaValue = 2 ** logAlphaValue
        
        allAUC = numpy.zeros(NUMBER_OF_FOLDS)
        allLogOutLL = numpy.zeros(NUMBER_OF_FOLDS)
        allHeldOutMisclassificationCosts = numpy.zeros(NUMBER_OF_FOLDS)
        
        for foldId, (train_index, eval_index) in enumerate(kFoldMaster.split(data, labels)):
            trainData = data[train_index]
            trainLabels = labels[train_index]
            evalData = data[eval_index]
            evalLabels = labels[eval_index]
            
            logReg = sklearn.linear_model.LogisticRegression(penalty="l2",C= 1.0 / alphaValue)
            logReg.fit(trainData, trainLabels)
            
            allAUC[foldId] = getTestAUC(logReg, evalData, evalLabels) 
            allLogOutLL[foldId] = getAverageHeldOutLogLikelihood(logReg, evalData, evalLabels)
            if misclassificationCosts is not None:
                predictedProbs = logReg.predict_proba(evalData)
                predictedLabels = bayesClassifier(predictedProbs, misclassificationCosts)
                allHeldOutMisclassificationCosts[foldId] = getAverageMisclassificationCosts(evalLabels, predictedLabels, misclassificationCosts) 
            
        allResults[regId, 0] = numpy.average(allHeldOutMisclassificationCosts)
        allResults[regId, 1] = numpy.average(allAUC)
        allResults[regId, 2] = numpy.average(allLogOutLL)
        
    if evalUsingLogLL:
        bestId = numpy.argmax(allResults[:,2])
        assert(allResults[bestId,2] > float("-inf") and allResults[bestId,2] < float("inf"))
    else:
        bestId = numpy.argmax(allResults[:,1])
        assert(allResults[bestId,1] > float("-inf") and allResults[bestId,1] < float("inf"))
    
    
    avgHoldOutMisclassificationCosts = allResults[bestId,0]
    
    logAlphaValue = lambdaRegValues[bestId]
    bestAlphaValue = 2 ** logAlphaValue
    logReg = sklearn.linear_model.LogisticRegression(penalty="l2",C= 1.0 / bestAlphaValue)
    logReg.fit(data, labels)
    
    return logReg, bestAlphaValue, avgHoldOutMisclassificationCosts




# mc-checked
def getBestNNModel(data, labels):
    
    evalUsingLogLL = True # MUST BE SET TO TRUE !!
    NUMBER_OF_FOLDS = 10
    kFoldMaster = sklearn.model_selection.StratifiedKFold(n_splits = NUMBER_OF_FOLDS,  shuffle=True, random_state=4324232)
    
    lambdaRegValues = numpy.arange(-10.0, 10.0, step = 0.5)
    lambdaRegValues = numpy.flip(lambdaRegValues, axis = 0)
    
    allResults = numpy.zeros((lambdaRegValues.shape[0], 3))
    
    for regId, logAlphaValue in enumerate(lambdaRegValues):
        alphaValue = 2 ** logAlphaValue
        
        allAUC = numpy.zeros(NUMBER_OF_FOLDS)
        allLogOutLL = numpy.zeros(NUMBER_OF_FOLDS)
        allHeldOutMisclassificationCosts = numpy.zeros(NUMBER_OF_FOLDS)
        
        for foldId, (train_index, eval_index) in enumerate(kFoldMaster.split(data, labels)):
            trainData = data[train_index]
            trainLabels = labels[train_index]
            evalData = data[eval_index]
            evalLabels = labels[eval_index]
            
            logReg = sklearn.linear_model.LogisticRegression(penalty="l2",C= 1.0 / alphaValue)
            logReg.fit(trainData, trainLabels)
            
            allAUC[foldId] = getTestAUC(logReg, evalData, evalLabels) 
            allLogOutLL[foldId] = getAverageHeldOutLogLikelihood(logReg, evalData, evalLabels)
            
            
        allResults[regId, 0] = numpy.average(allHeldOutMisclassificationCosts) 
        allResults[regId, 1] = numpy.average(allAUC)
        allResults[regId, 2] = numpy.average(allLogOutLL)
        
    if evalUsingLogLL:
        bestId = numpy.argmax(allResults[:,2])
        assert(allResults[bestId,2] > float("-inf") and allResults[bestId,2] < float("inf"))
    else:
        bestId = numpy.argmax(allResults[:,1])
        assert(allResults[bestId,1] > float("-inf") and allResults[bestId,1] < float("inf"))
    
    
    avgHoldOutMisclassificationCosts = allResults[bestId,0]
    
    logAlphaValue = lambdaRegValues[bestId]
    bestAlphaValue = 2 ** logAlphaValue
    logReg = sklearn.linear_model.LogisticRegression(penalty="l2",C= 1.0 / bestAlphaValue)
    logReg.fit(data, labels)
    
    return logReg, bestAlphaValue, avgHoldOutMisclassificationCosts




def getTrainedGAM_andBestLambda(data, labels):
    trainedGam = LogisticGAM()
    # trainedGam.gridsearch(data, labels, return_scores = True, lam = numpy.logspace(-5, 5, 20))
    trainedGam.gridsearch(data, labels)
    bestLambda = trainedGam.lam[0][0]    
    
    # this code is adapted from the source code of "gridsearch"
    newGam = deepcopy(trainedGam)
    newGam.set_params(trainedGam.get_params())
    newGam.set_params(coef_= trainedGam.coef_, force=True, verbose=False)
    newGam.set_params(lam = bestLambda)
    newGam.fit(data, labels)

    return newGam, bestLambda

#     print("new best lambda = ", bestLambda)
#     assert(False)
#     modelsToScores = trainedGam.gridsearch(data, labels, return_scores = True)
# 
#     bestModel = None
#     bestScore = numpy.inf
#     
#     for model, score in modelsToScores.items():
#         if score < bestScore:
#             bestModel = model
#             bestScore = score
#     
#     bestLambda = bestModel.lam[0][0]    
#     print("best lambda = ", bestLambda)
#     
#     trainedGam = LogisticGAM()
#     trainedGam.set_params(lam = bestLambda)
#     trainedGam.fit(data, labels)
#     return trainedGam, bestLambda

def ensure2D(predictedProbs):
    if len(predictedProbs.shape) < 2:
        # assume that predictedProbs is the true label prob
        allProbs = numpy.zeros((predictedProbs.shape[0],2))
        allProbs[:,0] = 1.0 - predictedProbs
        allProbs[:,1] = predictedProbs
         
        
        # print(allProbs)
        # assert(False)
        return allProbs
    else:
        return predictedProbs


def showPredctions(predicted_means, allSTDs, densityTestDataResponse):
    if allSTDs is not None:
        allLogPDF = scipy.stats.norm.logpdf(densityTestDataResponse, loc=predicted_means, scale=allSTDs)
        logProb = numpy.sum(allLogPDF)
    else:
        logProb = "-"
         
    mse = numpy.mean(numpy.square(predicted_means - densityTestDataResponse))

    # print("allLogPDF = ", allLogPDF)
    # print("allSTDs = ", allSTDs)
    # print("allMeansPredicted = ", allMeansPredicted)
    # print("densityTestDataResponse = ", densityTestDataResponse)
    
    print("logProb = ", logProb)
    print("mse = ", mse)
    return


def getBestGAM(data, labels, misclassificationCosts = None):
    
    trainedGam, bestLambda = getTrainedGAM_andBestLambda(data, labels)
    
    if misclassificationCosts is None:
        return trainedGam
    else:
        NUMBER_OF_FOLDS = 10
        kFoldMaster = sklearn.model_selection.StratifiedKFold(n_splits = NUMBER_OF_FOLDS,  shuffle=True, random_state=4324232)
    
        allHeldOutMisclassificationCosts = numpy.zeros(NUMBER_OF_FOLDS)
        
        noFailures = numpy.zeros(NUMBER_OF_FOLDS)
        
        for foldId, (train_index, eval_index) in enumerate(kFoldMaster.split(data, labels)):
            trainData = data[train_index]
            trainLabels = labels[train_index]
            evalData = data[eval_index]
            evalLabels = labels[eval_index]
            
            newGam = deepcopy(trainedGam)
            newGam.set_params(trainedGam.get_params())
            newGam.set_params(coef_= trainedGam.coef_, force=True, verbose=False)
            newGam.set_params(lam = bestLambda)
            try:
                newGam.fit(trainData, trainLabels)
            except ValueError as error:
                print("ERROR OCCURRED HERE !")
                print("trainData.shape = ", trainData.shape)
                print("trainLabels.shape = ", trainLabels)
                print("DETAILS:")
                print(error)
                continue
        
            predictedProbs = newGam.predict_proba(evalData)
            predictedProbs = ensure2D(predictedProbs)
            predictedLabels = bayesClassifier(predictedProbs, misclassificationCosts)
            allHeldOutMisclassificationCosts[foldId] = getAverageMisclassificationCosts(evalLabels, predictedLabels, misclassificationCosts)
            noFailures[foldId] = 1 
        
        assert(numpy.sum(noFailures) > 0)
        avgHoldOutMisclassificationCosts = numpy.sum(allHeldOutMisclassificationCosts[noFailures == 1]) / numpy.sum(noFailures)
        
        # print("avgHoldOutMisclassificationCosts = ", avgHoldOutMisclassificationCosts)
        # print("basic = ", numpy.average(allHeldOutMisclassificationCosts))
       
        return trainedGam, avgHoldOutMisclassificationCosts



def getPredictedLabelsAtThreshold(threshold, allPredictedProbs):
    assert(len(allPredictedProbs.shape) == 1)
    predictedLabels = numpy.zeros_like(allPredictedProbs)
    predictedLabels[allPredictedProbs >= threshold] = 1
    return predictedLabels

    
# reading checked
def getRecall(labels, allPredictedProbs, threshold):
    assert(numpy.all(numpy.logical_or(labels == 1, labels == 0)))
    
    totalTrueCount = numpy.sum(labels)
    coveredTrueCount = numpy.sum(labels[allPredictedProbs >= threshold])
    
    return coveredTrueCount / totalTrueCount



    
# reading checked
# calculate false discovery rate
def getFDR(labels, allPredictedProbs, threshold):
    assert(numpy.all(numpy.logical_or(labels == 1, labels == 0)))
    
    totalAlarmCount = (labels[allPredictedProbs >= threshold]).shape[0]
    truePositiveCount = numpy.sum(labels[allPredictedProbs >= threshold])
    
    assert(numpy.sum(allPredictedProbs >= threshold) == totalAlarmCount) # sanity check
    
    return 1.0 - (truePositiveCount / totalAlarmCount)



# reading checked
def getSpecifity(labels, allPredictedProbs, threshold):
    assert(numpy.all(numpy.logical_or(labels == 1, labels == 0)))
    
    labelsNeg = (labels - 1) * (-1)
    totalFalseCount = numpy.sum(labelsNeg)
    coveredFalseCount = numpy.sum(labelsNeg[allPredictedProbs < threshold])
    
    return coveredFalseCount / totalFalseCount


    
def getTestRecallSpecifityFDR(classificationModel, evalData, evalLabels, threshold):
    
    predictedProbTrueLabel = getPredictedProb(classificationModel, evalData)
    return getRecall(evalLabels, predictedProbTrueLabel, threshold), getSpecifity(evalLabels, predictedProbTrueLabel, threshold), getFDR(evalLabels, predictedProbTrueLabel, threshold)


# reading checked
def getPooledProbability(classificationModel_all, evalData):
    assert(len(classificationModel_all) == 5)
    allPredictedProbs = numpy.zeros((len(classificationModel_all), evalData.shape[0]))    
    
    for modelId in range(len(classificationModel_all)):
        predictedProb = classificationModel_all[modelId].predict_proba(evalData)
        allPredictedProbs[modelId, :] = predictedProb[:,1]
    
    allPredictedProbsPooled = numpy.mean(allPredictedProbs, axis = 0)
    
    assert(allPredictedProbsPooled.shape[0] == evalData.shape[0])
    return allPredictedProbsPooled



def getPooledRecallAndFDR(classificationModel_all, evalData, evalLabels, threshold_all, targetRecallId):
    assert(len(classificationModel_all) == 5)
    assert(len(threshold_all) == 5)
    allPredictedLabels = numpy.zeros((len(classificationModel_all),evalLabels.shape[0]))    
    
    for modelId in range(5):
        assert(len(threshold_all[modelId]) == 3)
        classificationModel = classificationModel_all[modelId]
        threshold = (threshold_all[modelId])[targetRecallId]
        predictedProbTrueLabel = classificationModel.predict_proba(evalData)[:,1]
        allPredictedLabels[modelId, predictedProbTrueLabel >= threshold] = 1
    
    
    predictedLabelsPooled = numpy.zeros(evalLabels.shape[0])
    predictedLabelsPooled[numpy.sum(allPredictedLabels, axis = 0) > 3] = 1  
    
    
    # print(numpy.sum(allPredictedLabels, axis = 0))
    # print(numpy.sum(allPredictedLabels, axis = 0).shape)
    # print("predictedLabelsPooled = ")
    # print(predictedLabelsPooled)
    
    totalTrueCount = numpy.sum(evalLabels)
    totalAlarmCount = (evalLabels[predictedLabelsPooled == 1]).shape[0]
    truePositiveCount = numpy.sum(evalLabels[predictedLabelsPooled == 1])
    FDR = 1.0 - (truePositiveCount / totalAlarmCount)
    recall = truePositiveCount / totalTrueCount

    return recall, FDR 


def getPredictedProb(classificationModel, evalData):
    predictedProb = classificationModel.predict_proba(evalData)
       
    assert(len(predictedProb.shape) == 1 or len(predictedProb.shape) == 2)
    if len(predictedProb.shape) == 2:
        predictedProbTrueLabel = predictedProb[:,1]
    else:
        predictedProbTrueLabel = predictedProb
        
    return predictedProbTrueLabel


def getFDR_atExactRecall(classificationModel, evalData, evalLabels, targetRecall):
    predictedProbTrueLabel = getPredictedProb(classificationModel, evalData)
    threshold = getThresholdFromPredictedProbabilities(evalLabels, predictedProbTrueLabel, targetRecall)
    return getRecall(evalLabels, predictedProbTrueLabel, threshold), getFDR(evalLabels, predictedProbTrueLabel, threshold)


# only for debugging
def getTestRecallDebug(classificationModel, evalData, evalLabels, threshold):
    predictedProbTrueLabel = getPredictedProb(classificationModel, evalData)
    
    y_pred = numpy.zeros_like(evalLabels)
    y_pred[predictedProbTrueLabel >= threshold] = 1
    print("scipy recall = ", sklearn.metrics.recall_score(evalLabels, y_pred))
    return

# reading checked
def getThresholdFromPredictedProbabilities(labels, allPredictedProbs, targetRecall):
    
    sortedIds = numpy.argsort(allPredictedProbs)
    sortedAllPredictedProbs = allPredictedProbs[sortedIds]
    sortedLabels = labels[sortedIds]
    
    totalTrueCount = numpy.sum(sortedLabels)
    
    previousThreshold = 0.0
    for i in range(sortedLabels.shape[0]):
        currentThreshold = sortedAllPredictedProbs[i]
        
        coveredTrueCount = numpy.sum(sortedLabels[i:sortedLabels.shape[0]])  
        recallEstimate = coveredTrueCount / totalTrueCount
          
        if recallEstimate < targetRecall:
            assert(previousThreshold <= currentThreshold) # it can be equal if two or more samples have the same true label probability
            
            myRecall = getRecall(labels, allPredictedProbs, previousThreshold) # sanity check
            assert(myRecall >= targetRecall)
             
            return previousThreshold
        
        previousThreshold = currentThreshold
        
    print("!! SOME ERROR OCCURRED !!")
    print("targetRecall = ", targetRecall)
    print("allPredictedProbs = ")
    print(allPredictedProbs)
    print("labels = ")
    print(labels)
    assert(False)
    
    return


# reading checked
def getThresholdEstimate(data, labels, bestHyperparameter, allTargetRecalls, modelType):
    assert(numpy.all(numpy.logical_or(labels == 1, labels == 0)))
    
    NUMBER_OF_FOLDS = 10
    kFoldMaster = sklearn.model_selection.StratifiedKFold(n_splits = NUMBER_OF_FOLDS,  shuffle=True, random_state=4324232)
    
    allPredictedProbs = -1.0 * numpy.ones_like(labels)
    
    for foldId, (train_index, eval_index) in enumerate(kFoldMaster.split(data, labels)):
        trainData = data[train_index]
        trainLabels = labels[train_index]
        evalData = data[eval_index]
        evalLabels = labels[eval_index]
        
        if modelType == "logReg":
            classificationModel = sklearn.linear_model.LogisticRegression(penalty="l2",C= 1.0 / bestHyperparameter)
            classificationModel.fit(trainData, trainLabels)
        elif modelType == "GP":
            covFuncRBF = sklearn.gaussian_process.kernels.RBF(length_scale = bestHyperparameter)
            classificationModel = sklearn.gaussian_process.GaussianProcessClassifier(kernel=covFuncRBF, optimizer = None)
            classificationModel.fit(trainData, trainLabels)
        elif modelType == "GAM":
            classificationModel = bestHyperparameter
        else:
            assert(False)
        
        predictedProbTrueLabel = getPredictedProb(classificationModel, evalData) 
        allPredictedProbs[eval_index] = predictedProbTrueLabel 
    
    assert(numpy.all(allPredictedProbs > -1))
    
    allThresholds = []
    for targetRecall in allTargetRecalls:
        threshold = getThresholdFromPredictedProbabilities(labels, allPredictedProbs, targetRecall)
        allThresholds.append(threshold)
    
    return allThresholds


def getThresholdEstimate_pooled(allTrainingData, labels, allBestHyperparameters, allTargetRecalls, modelType):
    assert(numpy.all(numpy.logical_or(labels == 1, labels == 0)))
    
    NUMBER_OF_FOLDS = 10
    kFoldMaster = sklearn.model_selection.StratifiedKFold(n_splits = NUMBER_OF_FOLDS,  shuffle=True, random_state=4324232)
    
    nrModels = len(allTrainingData)
    assert(len(allBestHyperparameters) == nrModels)
    assert(nrModels == 5)
    allPredictedProbs = -1.0 * numpy.ones((nrModels, labels.shape[0]))
    
    for foldId, (train_index, eval_index) in enumerate(kFoldMaster.split(allTrainingData[0], labels)):
        
        fixedEvalData = (allTrainingData[0])[eval_index]
        
        for modelId in range(nrModels):
            data = allTrainingData[modelId]
            bestHyperparameter = allBestHyperparameters[modelId]
            
            trainData = data[train_index]
            trainLabels = labels[train_index]
            
            
            if modelType == "logReg":
                classificationModel = sklearn.linear_model.LogisticRegression(penalty="l2",C= 1.0 / bestHyperparameter)
                classificationModel.fit(trainData, trainLabels)
            elif modelType == "GP":
                covFuncRBF = sklearn.gaussian_process.kernels.RBF(length_scale = bestHyperparameter)
                classificationModel = sklearn.gaussian_process.GaussianProcessClassifier(kernel=covFuncRBF, optimizer = None)
                classificationModel.fit(trainData, trainLabels)
            else:
                assert(False)
                
            predictedProb = classificationModel.predict_proba(fixedEvalData)
            allPredictedProbs[modelId, eval_index] = predictedProb[:,1] 
    
    assert(numpy.all(allPredictedProbs > -1))
    
    allPredictedProbsPooled = numpy.mean(allPredictedProbs, axis = 0) 
    
    allThresholds = []
    for targetRecall in allTargetRecalls:
        threshold = getThresholdFromPredictedProbabilities(labels, allPredictedProbsPooled, targetRecall)
        allThresholds.append(threshold)
    
    return allThresholds


# def getOracleBestL2RegularizedLogisticRegressionModel(trainData, trainLabels, testData, testLabels):
#     
#     lambdaRegValues = numpy.arange(-10.0, 10.0, step = 0.5)
#     lambdaRegValues = numpy.flip(lambdaRegValues, axis = 0)
#     
#     allResults = numpy.zeros((lambdaRegValues.shape[0], 3))
#     
#     for regId, logAlphaValue in enumerate(lambdaRegValues):
#         alphaValue = 2 ** logAlphaValue
#        
#         logReg = sklearn.linear_model.LogisticRegression(penalty="l2",C= 1.0 / alphaValue)
#         logReg.fit(trainData, trainLabels)
#         
#         predictedLabels = logReg.predict(testData)
#         allResults[regId, 0] = 0 
#         allResults[regId, 1] = sklearn.metrics.accuracy_score(testLabels, predictedLabels)
#         allResults[regId, 2] = getAverageHeldOutLogLikelihood(logReg, testData, testLabels)
#     
#     
#     logRegNoReg = sklearn.linear_model.LogisticRegression(penalty="l2",C= 2 ** 100)
#     logRegNoReg.fit(trainData, trainLabels)
#     noRegPerformance = sklearn.metrics.accuracy_score(testLabels, logRegNoReg.predict(testData))
#      
#     return numpy.max(allResults[:,1]), numpy.min(allResults[:,1]), noRegPerformance


# convert to -1/1 for greedy miser and AdaptGBRT
def getMatlabLabels(labels):
    matlabLabels = labels*2 - 1
    return matlabLabels
    

# ********************** prepare files for matlab - hyperparamer selection *******************************
def prepareForMatlab(trainData, trainLabels, evalData, evalLabels, featureCosts, outputFilename, classificationModelName):
    assert(numpy.all(numpy.logical_or(evalLabels == 0, evalLabels == 1)))
     
    NUMBER_OF_FEATURES = trainData.shape[1]
    SIZE_OF_TRAIN_DATA = trainData.shape[0]
    SIZE_OF_EVAL_DATA = evalData.shape[0]
    print("NUMBER_OF_FEATURES = ", NUMBER_OF_FEATURES)
    print("SIZE_OF_TRAIN_DATA = ", SIZE_OF_TRAIN_DATA)
    print("SIZE_OF_EVAL_DATA = ", SIZE_OF_EVAL_DATA)
    
    
    bestModel, _ = getBestModel(trainData, trainLabels, classificationModelName)
    # bestModel, _, _ = getBestL2RegularizedLogisticRegressionModelNew(trainData, trainLabels)
    
    # print("bestModel.predict_proba(trainData) = ", ensure2D(bestModel.predict_proba(trainData)))
    # print("trainingTrueProbs = ", trainingTrueProbs)
    # assert(False)
    # print("before = ",evalLabels)
    # print("getMatlabLabels(evalLabels) = ", getMatlabLabels(evalLabels))
    # assert(False)
    
    matlabDict = {}
    matlabDict["xtr"] = trainData
    matlabDict["xte"] = evalData
    matlabDict["ytr"] = numpy.asmatrix(getMatlabLabels(trainLabels), dtype = numpy.double).transpose()
    matlabDict["yte"] = numpy.asmatrix(getMatlabLabels(evalLabels), dtype = numpy.double).transpose()
    matlabDict["cost"] = numpy.asmatrix(featureCosts, dtype = numpy.double).transpose()
    
    matlabDict["proba_pred_train"] = ensure2D(bestModel.predict_proba(trainData))
    matlabDict["proba_pred_test"] = ensure2D(bestModel.predict_proba(evalData))
    matlabDict["feature_usage_test"] = numpy.ones((SIZE_OF_EVAL_DATA,NUMBER_OF_FEATURES))
    
    scipy.io.savemat(outputFilename + "_" + classificationModelName, matlabDict)
    
    print("FINISHED PREPARING ALL MATLAB FILES FOR FINDING HYPER-PARAMETERS")
    return


def getTestAccuracy(classificationModel, evalData, evalLabels):
    predictedLabels = classificationModel.predict(evalData)
    averageAccuracy = sklearn.metrics.accuracy_score(evalLabels, predictedLabels)
    return averageAccuracy


def getTestAUC(classificationModel, evalData, evalLabels):
    predictedProbTrueLabel = getPredictedProb(classificationModel, evalData)
    
    auc = sklearn.metrics.roc_auc_score(evalLabels, predictedProbTrueLabel)
    return auc




# mc-checked
def getBestGP(data, labels):
    
    rndIds = numpy.arange(labels.shape[0])
    numpy.random.shuffle(rndIds)
    trainDataSize = int(0.9 * labels.shape[0])
    train_index = rndIds[0:trainDataSize]
    eval_index = rndIds[trainDataSize:rndIds.shape[0]]
    
    allLengthScaleValues = numpy.linspace(0.001, 10.0, num = 20)
    # print("lengthScaleValue = ", allLengthScaleValues)
        
    allScores = numpy.zeros_like(allLengthScaleValues)
    for i, lengthScaleValue in enumerate(allLengthScaleValues):
        trainData = data[train_index]
        trainLabels = labels[train_index]
        evalData = data[eval_index]
        evalLabels = labels[eval_index]
                
        covFuncRBF = sklearn.gaussian_process.kernels.RBF(length_scale = lengthScaleValue)
        gpClassifier = sklearn.gaussian_process.GaussianProcessClassifier(kernel=covFuncRBF, optimizer = None)
        
        # covFuncConst = sklearn.gaussian_process.kernels.ConstantKernel(constant_value=1.0)
        # covFuncFinal = sklearn.gaussian_process.kernels.Product(covFuncConst, covFuncRBF)
        # gpClassifier = sklearn.gaussian_process.GaussianProcessClassifier(kernel=covFuncFinal)
        gpClassifier.fit(trainData, trainLabels)
        
        # allScores[i] = getTestAUC(gpClassifier, evalData, evalLabels)
        # print("train auc = ", getTestAUC(gpClassifier, trainData, trainLabels))
        # print("eval auc = ", getTestAUC(gpClassifier, evalData, evalLabels))
        allScores[i] = getAverageHeldOutLogLikelihood(gpClassifier, evalData, evalLabels)
        
    
    # print("best eval auc = ", numpy.max(allScores))
    bestLengthScale = allLengthScaleValues[numpy.argmax(allScores)]
    print("bestLengthScale = ", bestLengthScale)
    
    # final training on all data
    covFuncRBF = sklearn.gaussian_process.kernels.RBF(length_scale = bestLengthScale)
    gpClassifier = sklearn.gaussian_process.GaussianProcessClassifier(kernel=covFuncRBF, optimizer = None)
    gpClassifier.fit(data, labels)
    
    return gpClassifier, bestLengthScale



# mc-checked
def getBestSVMRBFModel(data, labels, misclassificationCosts):
    NUMBER_OF_FOLDS = 5
    kFoldMaster = sklearn.model_selection.StratifiedKFold(n_splits = NUMBER_OF_FOLDS,  shuffle=True, random_state=4324232)
        
    C_exp_values = numpy.arange(-5.0, 15.0, step = 0.5)
    gamma_exp_values = numpy.arange(-15.0, 3.0, step = 0.5)
    
    allResultsMC = numpy.zeros((C_exp_values.shape[0], gamma_exp_values.shape[0]))
    allResultsACC = numpy.zeros((C_exp_values.shape[0], gamma_exp_values.shape[0]))
    
    for cId, cExp in enumerate(C_exp_values):
        for gammaId, gammaExp in enumerate(gamma_exp_values):
            cValue = 2 ** cExp
            gammaValue = 2 ** gammaExp
        
            allMisclassificationCosts = numpy.zeros(NUMBER_OF_FOLDS)
            allAccuracies = numpy.zeros(NUMBER_OF_FOLDS)
       
            for foldId, (train_index, eval_index) in enumerate(kFoldMaster.split(data, labels)):
                trainData = data[train_index]
                trainLabels = labels[train_index]
                evalData = data[eval_index]
                evalLabels = labels[eval_index]
                
                svmRbf = sklearn.svm.SVC(C=cValue, kernel='rbf', gamma=gammaValue) 
                svmRbf.fit(trainData, trainLabels)
                predictedLabels = svmRbf.predict(evalData)
                allMisclassificationCosts[foldId] = getAverageMisclassificationCosts(evalLabels, predictedLabels, misclassificationCosts)
                allAccuracies[foldId] = sklearn.metrics.accuracy_score(evalLabels, predictedLabels)
            
            allResultsMC[cId, gammaId] = numpy.average(allMisclassificationCosts)
            allResultsACC[cId, gammaId] = numpy.average(allAccuracies)
        
    
    
    # bestIds = numpy.argmax(allResultsACC)
    bestIds = numpy.unravel_index(numpy.argmax(allResultsACC, axis=None), allResultsACC.shape)
    
    # print("allResultsACC = ")
    # print(allResultsACC)
    # print("bestIds = ", bestIds)
    print("best validation accuracy = ", allResultsACC[bestIds])
    
    cValue = 2 ** C_exp_values[bestIds[0]]
    gammaValue = 2 ** gamma_exp_values[bestIds[1]]
    
    svmRbf = sklearn.svm.SVC(C=cValue, kernel='rbf', gamma=gammaValue) 
    svmRbf.fit(data, labels)
    return svmRbf        

# mc-checked
def getBestSVMRBFModelAcc(data, labels):
    NUMBER_OF_FOLDS = 5
    kFoldMaster = sklearn.model_selection.StratifiedKFold(n_splits = NUMBER_OF_FOLDS,  shuffle=True, random_state=4324232)
        
    C_exp_values = numpy.arange(-5.0, 15.0, step = 0.5)
    gamma_exp_values = numpy.arange(-15.0, 3.0, step = 0.5)
    
    allResultsACC = numpy.zeros((C_exp_values.shape[0], gamma_exp_values.shape[0]))
    
    for cId, cExp in enumerate(C_exp_values):
        for gammaId, gammaExp in enumerate(gamma_exp_values):
            cValue = 2 ** cExp
            gammaValue = 2 ** gammaExp
        
            allAccuracies = numpy.zeros(NUMBER_OF_FOLDS)
       
            for foldId, (train_index, eval_index) in enumerate(kFoldMaster.split(data, labels)):
                trainData = data[train_index]
                trainLabels = labels[train_index]
                evalData = data[eval_index]
                evalLabels = labels[eval_index]
                
                svmRbf = sklearn.svm.SVC(C=cValue, kernel='rbf', gamma=gammaValue) 
                svmRbf.fit(trainData, trainLabels)
                predictedLabels = svmRbf.predict(evalData)
                allAccuracies[foldId] = sklearn.metrics.accuracy_score(evalLabels, predictedLabels)
            
            allResultsACC[cId, gammaId] = numpy.average(allAccuracies)
        
    
    
    bestIds = numpy.unravel_index(numpy.argmax(allResultsACC, axis=None), allResultsACC.shape)
    
    print("best validation accuracy = ", allResultsACC[bestIds])
    
    cValue = 2 ** C_exp_values[bestIds[0]]
    gammaValue = 2 ** gamma_exp_values[bestIds[1]]
    
    svmRbf = sklearn.svm.SVC(C=cValue, kernel='rbf', gamma=gammaValue) 
    svmRbf.fit(data, labels)
    return svmRbf        

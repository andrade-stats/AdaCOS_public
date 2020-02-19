import numpy
import experimentSetting
import experimentHelper
import evaluation
import scipy.io
import os

import constants
import realdata
import sklearn.metrics

# dataName = "breastcancer_5foldCV"
# dataName = "pima_5foldCV"
# dataName = "pyhsioNetWithMissing_5foldCV"
dataName = "heartDiseaseWithMissing_5foldCV"







resultsRecorder = experimentHelper.ResultsRecorder(len(constants.allFalsePositiveCosts))
readyForSaving = False

definedFeatureCosts = realdata.getFeaturesCosts(dataName)

for falsePositiveCost in constants.allFalsePositiveCosts:
    
    falseNegativeCost = falsePositiveCost * constants.FN_TO_FP_RATIO
    allBestSettings = numpy.zeros((constants.NUMBER_OF_FOLDS, 2), dtype = numpy.int)
    
    for testFoldId in range(constants.NUMBER_OF_FOLDS):
    
        print("*************************** GREEDY MISER *******************************************")
         
        bestLambdaId, bestTreeId = evaluation.getBestParametersForGreedyMiser_asymmetric(dataName, definedFeatureCosts, testFoldId, falsePositiveCost, falseNegativeCost)
       
        allBestSettings[testFoldId, 0] = bestLambdaId
        allBestSettings[testFoldId, 1] = bestTreeId
    
    
    # save best settings from validation data
    outputFilename = experimentSetting.MATLAB_FOLDER_BEST_SETTINGS_GREEDY_MISER + dataName + "_" + str(int(falsePositiveCost)) + "_allBestSettings_" + "asymmetric" 
    matlabDict = {}
    matlabDict["allBestSettings"] = numpy.asmatrix(allBestSettings, dtype = numpy.int)
    scipy.io.savemat(outputFilename, matlabDict)
    
    
    if os.path.isfile(experimentSetting.MATLAB_FOLDER_RESULTS_GREEDY_MISER + dataName + "_" + str(int(falsePositiveCost)) + "_forFinalTrainingAndTesting_" + str(4) + "_allResults_" + "asymmetric"  + ".mat"):
    
        print("*************************** AFTER FINAL TRAINING *******************************************")
        
        testTotalCostsAllFolds = numpy.zeros(constants.NUMBER_OF_FOLDS)
        testFeatureCostsAllFolds = numpy.zeros(constants.NUMBER_OF_FOLDS)
        testMisClassificationCostsAllFolds = numpy.zeros(constants.NUMBER_OF_FOLDS)
        testAccuracyAllFolds = numpy.zeros(constants.NUMBER_OF_FOLDS)
        testAUCAllFolds = numpy.zeros(constants.NUMBER_OF_FOLDS)
        
        testRecallAllFolds = numpy.zeros(constants.NUMBER_OF_FOLDS)
        testFDRAllFolds = numpy.zeros(constants.NUMBER_OF_FOLDS)
        testOperationCostsAllFolds = numpy.zeros(constants.NUMBER_OF_FOLDS)
        
        testRecallAllFolds_exactRecall = numpy.zeros(constants.NUMBER_OF_FOLDS)
        testFDRAllFolds_exactRecall = numpy.zeros(constants.NUMBER_OF_FOLDS)
        testOperationCostsAllFolds_exactRecall = numpy.zeros(constants.NUMBER_OF_FOLDS)
    
        for testFoldId in range(constants.NUMBER_OF_FOLDS):
         
            allResultsInMatlab = scipy.io.loadmat(experimentSetting.MATLAB_FOLDER_RESULTS_GREEDY_MISER + dataName + "_" + str(int(falsePositiveCost)) + "_forFinalTrainingAndTesting_" + str(testFoldId) + "_allResults_" + "asymmetric" )
            avgFeatureCosts_allTrees = (allResultsInMatlab['allTotalCost'].transpose())[0]
            scores_allTrees = allResultsInMatlab['allScores']
            
            assert(avgFeatureCosts_allTrees.shape[0] == scores_allTrees.shape[1])
            
            bestTreeId = allBestSettings[testFoldId, 1]
            predictedTestLabels = evaluation.getLabelsFromGreedyScores(scores_allTrees[:,bestTreeId])
            predictedTestTrueLabelProbs = evaluation.getProbabilitiesFromGreedyScores(scores_allTrees[:,bestTreeId])
            avgTestFeatureCosts = avgFeatureCosts_allTrees[bestTreeId]
                
            _, _, _, _, testLabels = realdata.loadSubset(dataName, None, testFoldId, constants.IMPUTATION_METHOD)
            
            assert(avgTestFeatureCosts <= numpy.sum(definedFeatureCosts)) # just to ensure that it is really the average and not a sum over all samples
                       
            # testOperationCostsAllFolds_exactRecall[testFoldId], testFDRAllFolds_exactRecall[testFoldId], testRecallAllFolds_exactRecall[testFoldId] = evaluation.getResultsAtTargetRecall(falsePositiveCost, targetRecall, testLabels, predictedTestTrueLabelProbs, avgTestFeatureCosts) 
            # threshold_forExactRecall = evaluation.getThresholdFromPredictedProbabilities(testLabels, predictedTestTrueLabelProbs, targetRecall)
            # testRecallAllFolds_exactRecall[testFoldId] = evaluation.getRecall(testLabels, predictedTestTrueLabelProbs, threshold_forExactRecall)
            # testFDRAllFolds_exactRecall[testFoldId] = evaluation.getFDR(testLabels, predictedTestTrueLabelProbs, threshold_forExactRecall)
            # predictedTestLabels_atExactRecall = evaluation.getPredictedLabelsAtThreshold(threshold_forExactRecall, predictedTestTrueLabelProbs)
            # testOperationCostsAllFolds_exactRecall[testFoldId] = evaluation.getAverageOperationCosts(testLabels, predictedTestLabels_atExactRecall, avgTestFeatureCosts, falsePositiveCost)
            
            testFeatureCostsAllFolds[testFoldId] = avgTestFeatureCosts
            
            misclassificationCosts = numpy.zeros((2, 2))
            misclassificationCosts[0, 1] = falsePositiveCost 
            misclassificationCosts[1, 0] = falseNegativeCost 
            misclassificationCosts[0, 0] = 0.0
            misclassificationCosts[1, 1] = 0.0
            
            testTotalCostsAllFolds[testFoldId] = evaluation.getAverageTotalCosts(testLabels, predictedTestLabels, avgTestFeatureCosts, misclassificationCosts)
            testMisClassificationCostsAllFolds[testFoldId] = evaluation.getAverageMisclassificationCosts(testLabels, predictedTestLabels, misclassificationCosts)
            testAccuracyAllFolds[testFoldId] = sklearn.metrics.accuracy_score(testLabels, predictedTestLabels)
            testAUCAllFolds[testFoldId] = sklearn.metrics.roc_auc_score(testLabels, predictedTestTrueLabelProbs)
        
            testRecallAllFolds[testFoldId] = sklearn.metrics.recall_score(testLabels, predictedTestLabels)
            testFDRAllFolds[testFoldId] = 1.0 - sklearn.metrics.precision_score(testLabels, predictedTestLabels)
            testOperationCostsAllFolds[testFoldId] = evaluation.getAverageOperationCosts(testLabels, predictedTestLabels, avgTestFeatureCosts, falsePositiveCost)
            
            
        resultsRecorder.addAll(falsePositiveCost, (testTotalCostsAllFolds, testFeatureCostsAllFolds, testMisClassificationCostsAllFolds, testAccuracyAllFolds, testAUCAllFolds, testRecallAllFolds, testFDRAllFolds, testOperationCostsAllFolds, testRecallAllFolds_exactRecall, testFDRAllFolds_exactRecall, testOperationCostsAllFolds_exactRecall))
        readyForSaving = True
    

   
if readyForSaving:
    filename = dataName + "_GreedyMiser_" + "asymmetricCost"
    resultsRecorder.writeOutResults(filename)
    print("WROTE OUT FINAL RESULTS INTO: " + filename)

print("Finished Successfully")
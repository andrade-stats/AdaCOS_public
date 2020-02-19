import numpy
import experimentSetting
import experimentHelper
import evaluation
import scipy.io
import os

import constants
import realdata
import sklearn.metrics

dataName = "pima_5foldCV"
# dataName = "breastcancer_5foldCV"
# dataName = "pyhsioNetWithMissing_5foldCV"
# dataName = "heartDiseaseWithMissing_5foldCV"




imputationMethod = "gaussian_imputation"


targetRecall = 0.99




NUMBER_OF_FOLDS = 5

resultsRecorder = experimentHelper.ResultsRecorder(len(constants.allFalsePositiveCosts))
readyForSaving = False

definedFeatureCosts = realdata.getFeaturesCosts(dataName)

for falsePositiveCost in constants.allFalsePositiveCosts:
    
    allBestSettings = numpy.zeros((NUMBER_OF_FOLDS, 2), dtype = numpy.int)
    allThresholds = numpy.zeros(NUMBER_OF_FOLDS)
     
    for testFoldId in range(NUMBER_OF_FOLDS):
    
        print("*************************** GREEDY MISER *******************************************")
         
        # use USE_AVG_TREES = False since the results are better
        bestLambdaId, bestTreeId, threshold = evaluation.getBestParametersForGreedyMiser_targetRecall(dataName, definedFeatureCosts, testFoldId, falsePositiveCost, targetRecall, averageTrees = False) 
       
        allBestSettings[testFoldId, 0] = bestLambdaId
        allBestSettings[testFoldId, 1] = bestTreeId
        allThresholds[testFoldId] = threshold
    
    
    # save best settings from validation data
    outputFilename = experimentSetting.MATLAB_FOLDER_BEST_SETTINGS_GREEDY_MISER + dataName + "_" + str(int(falsePositiveCost)) + "_allBestSettings_" + str(targetRecall) + "targetRecall" 
    matlabDict = {}
    matlabDict["allBestSettings"] = numpy.asmatrix(allBestSettings, dtype = numpy.int)
    scipy.io.savemat(outputFilename, matlabDict)
    
    
    if os.path.isfile(experimentSetting.MATLAB_FOLDER_RESULTS_GREEDY_MISER + dataName + "_" + str(int(falsePositiveCost)) + "_forFinalTrainingAndTesting_" + str(4) + "_allResults_" + str(targetRecall) + "targetRecall"  + ".mat"):
    
        print("*************************** AFTER FINAL TRAINING *******************************************")
        
        testTotalCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
        testFeatureCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
        testMisClassificationCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
        testAccuracyAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
        testAUCAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
        
        testRecallAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
        testFDRAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
        testOperationCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
        
        testRecallAllFolds_exactRecall = numpy.zeros(NUMBER_OF_FOLDS)
        testFDRAllFolds_exactRecall = numpy.zeros(NUMBER_OF_FOLDS)
        testOperationCostsAllFolds_exactRecall = numpy.zeros(NUMBER_OF_FOLDS)
    
        for testFoldId in range(NUMBER_OF_FOLDS):
         
            allResultsInMatlab = scipy.io.loadmat(experimentSetting.MATLAB_FOLDER_RESULTS_GREEDY_MISER + dataName + "_" + str(int(falsePositiveCost)) + "_forFinalTrainingAndTesting_" + str(testFoldId) + "_allResults_" + str(targetRecall) + "targetRecall" )
            avgFeatureCosts_allTrees = (allResultsInMatlab['allTotalCost'].transpose())[0]
            scores_allTrees = allResultsInMatlab['allScores']
            
            assert(avgFeatureCosts_allTrees.shape[0] == scores_allTrees.shape[1])
            
            bestTreeId = allBestSettings[testFoldId, 1]
            predictedTestTrueLabelProbs = evaluation.getProbabilitiesFromGreedyScores(scores_allTrees[:,bestTreeId])
            avgTestFeatureCosts = avgFeatureCosts_allTrees[bestTreeId]
                
            _, _, _, _, testLabels = realdata.loadSubset(dataName, None, testFoldId, imputationMethod)
            
            assert(avgTestFeatureCosts <= numpy.sum(definedFeatureCosts)) # just to ensure that it is really the average and not a sum over all samples
            
            threshold = allThresholds[testFoldId]
            
            # set to same recall as proposed method to allow for fair comparison
            targetRecall_fromProposedMethod = evaluation.getTargetRecallFromProposedMethod(dataName, falsePositiveCost, targetRecall)
            threshold_forExactRecall = evaluation.getThresholdFromPredictedProbabilities(testLabels, predictedTestTrueLabelProbs, targetRecall_fromProposedMethod)
            testRecallAllFolds_exactRecall[testFoldId] = evaluation.getRecall(testLabels, predictedTestTrueLabelProbs, threshold_forExactRecall)
            testFDRAllFolds_exactRecall[testFoldId] = evaluation.getFDR(testLabels, predictedTestTrueLabelProbs, threshold_forExactRecall)
            
            predictedTestLabels_atExactRecall = evaluation.getPredictedLabelsAtThreshold(threshold_forExactRecall, predictedTestTrueLabelProbs)
            testOperationCostsAllFolds_exactRecall[testFoldId] = evaluation.getAverageOperationCosts(testLabels, predictedTestLabels_atExactRecall, avgTestFeatureCosts, falsePositiveCost)
            
            testFeatureCostsAllFolds[testFoldId] = avgTestFeatureCosts
            
            predictedTestLabels = evaluation.getPredictedLabelsAtThreshold(threshold, predictedTestTrueLabelProbs)
            testAccuracyAllFolds[testFoldId] = sklearn.metrics.accuracy_score(testLabels, predictedTestLabels)
            testAUCAllFolds[testFoldId] = sklearn.metrics.roc_auc_score(testLabels, predictedTestTrueLabelProbs)
        
            testRecallAllFolds[testFoldId] = sklearn.metrics.recall_score(testLabels, predictedTestLabels)
            testFDRAllFolds[testFoldId] = 1.0 - sklearn.metrics.precision_score(testLabels, predictedTestLabels)
            testOperationCostsAllFolds[testFoldId] = evaluation.getAverageOperationCosts(testLabels, predictedTestLabels, avgTestFeatureCosts, falsePositiveCost)
            
            
        resultsRecorder.addAll(falsePositiveCost, (testTotalCostsAllFolds, testFeatureCostsAllFolds, testMisClassificationCostsAllFolds, testAccuracyAllFolds, testAUCAllFolds, testRecallAllFolds, testFDRAllFolds, testOperationCostsAllFolds, testRecallAllFolds_exactRecall, testFDRAllFolds_exactRecall, testOperationCostsAllFolds_exactRecall))
        readyForSaving = True
    

   
if readyForSaving:
    filename = dataName + "_GreedyMiser_" + str(targetRecall) + "recall"
    resultsRecorder.writeOutResults(filename)
    print("WROTE OUT FINAL RESULTS INTO: " + filename)

print("Finished Successfully")
import numpy
import evaluation
import sklearn.metrics
import realdata

import experimentHelper
import constants

def getAllStatistics(filename):
    allInfoArray = numpy.load(filename)
    numberOfCovariates = allInfoArray.shape[0] - 2
    covariateUsage = allInfoArray[:, 2:allInfoArray.shape[0]]
    
    labels = allInfoArray[:,0]
    labels = labels.astype(numpy.int)
    predictedProbTrueLabel = allInfoArray[:,1]
    return labels, predictedProbTrueLabel, covariateUsage

def getAvgCovariateCosts(definedFeatureCosts, covariateUsage):
    totalCovariateCostsEachSample = numpy.sum((covariateUsage * definedFeatureCosts), axis = 1)
    return numpy.mean(totalCovariateCostsEachSample)





dataName = "pima_5foldCV"
# dataName = "breastcancer_5foldCV"
# dataName = "pyhsioNetWithMissing_5foldCV"
# dataName = "heartDiseaseWithMissing_5foldCV"


# COST_TYPE = "asymmetricCost"
# COST_TYPE = "symmetricCost"
COST_TYPE = "recall"
targetRecall = 0.99


# print("test = "  + str(targetRecall) + COST_TYPE)
# assert(False)

allRCOSTS = numpy.linspace(start=0.0, stop=-0.1, num = 11)



BASEFOLDER = "/export/home/s-andrade/newStart/dynamicCovariateBaselines/Joint-AFA-Classification_modified/results/"
NUMBER_OF_FOLDS = 5


definedFeatureCosts = realdata.getFeaturesCosts(dataName)

if COST_TYPE == "symmetricCost":
    ALL_FALSE_POSITIVE_COSTS = [400,800]
else:
    ALL_FALSE_POSITIVE_COSTS = constants.allFalsePositiveCosts


    

resultsRecorder = experimentHelper.ResultsRecorder(len(ALL_FALSE_POSITIVE_COSTS))

for falsePositiveCost in ALL_FALSE_POSITIVE_COSTS:
    print("")
    print("falsePositiveCost = ", falsePositiveCost)

    testFeatureCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    testRecallAllFolds_exactRecall = numpy.zeros(NUMBER_OF_FOLDS)
    testFDRAllFolds_exactRecall = numpy.zeros(NUMBER_OF_FOLDS)
    testOperationCostsAllFolds_exactRecall = numpy.zeros(NUMBER_OF_FOLDS)
    
    testRecallAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    testFDRAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    testOperationCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    
    testTotalCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    testMisClassificationCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    testAccuracyAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    testAUCAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    
    for foldId in range(NUMBER_OF_FOLDS):
        
        if COST_TYPE == "recall":
            allOperationsCostsValidation = numpy.zeros(len(allRCOSTS))
            allThresholds = numpy.zeros(len(allRCOSTS))
            for penaltyId, featureCostsPenalty in enumerate(allRCOSTS):
                FILENAME_STEM = BASEFOLDER + dataName + "_fold" + str(foldId) + "/" + "costs" + str(featureCostsPenalty) + "_"
                validationLabels, allPredictedProbsValidation, covariateUsageValidation = getAllStatistics(FILENAME_STEM + "val.npy")
                avgValidationFeatureCosts = getAvgCovariateCosts(definedFeatureCosts, covariateUsageValidation)
                
                thresholdValidation = evaluation.getThresholdFromPredictedProbabilities(validationLabels, allPredictedProbsValidation, targetRecall)
                predictedValidationLabels = evaluation.getPredictedLabelsAtThreshold(thresholdValidation, allPredictedProbsValidation)
                allOperationsCostsValidation[penaltyId] = evaluation.getAverageOperationCosts(validationLabels, predictedValidationLabels, avgValidationFeatureCosts, falsePositiveCost)
                allThresholds[penaltyId] = thresholdValidation
            
            bestPenaltyId = numpy.argmin(allOperationsCostsValidation)
            bestFeatureCostsPenalty = allRCOSTS[bestPenaltyId]       
            thresholdTest = allThresholds[bestPenaltyId]
            
            FILENAME_STEM = BASEFOLDER + dataName + "_fold" + str(foldId) + "/" + "costs" + str(bestFeatureCostsPenalty) + "_"
            testLabels, allPredictedProbsTest, covariateUsageTest = getAllStatistics(FILENAME_STEM + "ts.npy")
            predictedTestLabels = evaluation.getPredictedLabelsAtThreshold(thresholdTest, allPredictedProbsTest)
           
            avgTestFeatureCosts = getAvgCovariateCosts(definedFeatureCosts, covariateUsageTest)
            
            testFeatureCostsAllFolds[foldId] = avgTestFeatureCosts
            testAccuracyAllFolds[foldId] = sklearn.metrics.accuracy_score(testLabels, predictedTestLabels)
            testAUCAllFolds[foldId] = sklearn.metrics.roc_auc_score(testLabels, allPredictedProbsTest)
        
            testRecallAllFolds[foldId] = sklearn.metrics.recall_score(testLabels, predictedTestLabels)
            testFDRAllFolds[foldId] = 1.0 - sklearn.metrics.precision_score(testLabels, predictedTestLabels)
            testOperationCostsAllFolds[foldId] = evaluation.getAverageOperationCosts(testLabels, predictedTestLabels, avgTestFeatureCosts, falsePositiveCost)
            
            # set to same recall as proposed method to allow for fair comparison
            targetRecall_fromProposedMethod = evaluation.getTargetRecallFromProposedMethod(dataName, falsePositiveCost, targetRecall)
            threshold_forExactRecall = evaluation.getThresholdFromPredictedProbabilities(testLabels, allPredictedProbsTest, targetRecall_fromProposedMethod)
            predictedTestLabels_atExactRecall = evaluation.getPredictedLabelsAtThreshold(threshold_forExactRecall, allPredictedProbsTest)
            testRecallAllFolds_exactRecall[foldId] = sklearn.metrics.recall_score(testLabels, predictedTestLabels_atExactRecall)
            testFDRAllFolds_exactRecall[foldId] = 1.0 - sklearn.metrics.precision_score(testLabels, predictedTestLabels_atExactRecall)
            testOperationCostsAllFolds_exactRecall[foldId] = evaluation.getAverageOperationCosts(testLabels, predictedTestLabels_atExactRecall, avgTestFeatureCosts, falsePositiveCost)
            
        else:
            
            if COST_TYPE == "symmetricCost":
                assert(dataName == "pima_5foldCV")
                falseNegativeCost = falsePositiveCost
                correctClassificationCost = -50.0 # in order to align to the setting in (Ji and Carin, 2007; Dulac-Arnold et al., 2012) for Diabetes.
            elif COST_TYPE == "asymmetricCost":
                falseNegativeCost = falsePositiveCost * constants.FN_TO_FP_RATIO
                correctClassificationCost = 0.0 
            else:
                assert(False)
                
            misclassificationCosts = numpy.zeros((2, 2))
            misclassificationCosts[0, 1] = falsePositiveCost 
            misclassificationCosts[1, 0] = falseNegativeCost 
            misclassificationCosts[0, 0] = correctClassificationCost
            misclassificationCosts[1, 1] = correctClassificationCost
            
            allTotalCostsValidation = numpy.zeros(len(allRCOSTS))
            for penaltyId, featureCostsPenalty in enumerate(allRCOSTS):
                FILENAME_STEM = BASEFOLDER + dataName + "_fold" + str(foldId) + "/" + "costs" + str(featureCostsPenalty) + "_"
                validationLabels, allPredictedProbsValidation, covariateUsageValidation = getAllStatistics(FILENAME_STEM + "val.npy")
                avgValidationFeatureCosts = getAvgCovariateCosts(definedFeatureCosts, covariateUsageValidation)
                
                predictedValidationLabels = evaluation.bayesClassifier(allPredictedProbsValidation, misclassificationCosts)
                
                allTotalCostsValidation[penaltyId] = evaluation.getAverageTotalCosts(validationLabels, predictedValidationLabels, avgValidationFeatureCosts, misclassificationCosts)
            
            bestPenaltyId = numpy.argmin(allTotalCostsValidation)
            bestFeatureCostsPenalty = allRCOSTS[bestPenaltyId]
            
            FILENAME_STEM = BASEFOLDER + dataName + "_fold" + str(foldId) + "/" + "costs" + str(bestFeatureCostsPenalty) + "_"
            testLabels, allPredictedProbsTest, covariateUsageTest = getAllStatistics(FILENAME_STEM + "ts.npy")
            avgTestFeatureCosts = getAvgCovariateCosts(definedFeatureCosts, covariateUsageTest)
            
            predictedTestLabels = evaluation.bayesClassifier(allPredictedProbsTest, misclassificationCosts)
                
            testTotalCostsAllFolds[foldId] = evaluation.getAverageTotalCosts(testLabels, predictedTestLabels, avgTestFeatureCosts, misclassificationCosts)
            
            testFeatureCostsAllFolds[foldId] = avgTestFeatureCosts
            testMisClassificationCostsAllFolds[foldId] = evaluation.getAverageMisclassificationCosts(testLabels, predictedTestLabels, misclassificationCosts)
            testAccuracyAllFolds[foldId] = sklearn.metrics.accuracy_score(testLabels, predictedTestLabels)
            testAUCAllFolds[foldId] = sklearn.metrics.roc_auc_score(testLabels, allPredictedProbsTest)
        
            testRecallAllFolds[foldId] = sklearn.metrics.recall_score(testLabels, predictedTestLabels)
            testFDRAllFolds[foldId] = 1.0 - sklearn.metrics.precision_score(testLabels, predictedTestLabels)
            testOperationCostsAllFolds[foldId] = evaluation.getAverageOperationCosts(testLabels, predictedTestLabels, avgTestFeatureCosts, falsePositiveCost)
            
        
    # evaluation.showHelperDetailed("FDR = ", testFDRAllFolds_exactRecall)
    # evaluation.showHelperDetailed("OPC = ", testOperationCostsAllFolds_exactRecall)
    # evaluation.showHelperDetailed("costs of acquired features = ", testFeatureCostsAllFolds)
    # evaluation.showHelperDetailed("recall = ", testRecallAllFolds_exactRecall)

    # allFDRstr.append(evaluation.getDetailedStr(testFDRAllFolds_exactRecall))
    # allOPCstr.append(evaluation.getDetailedStr(testOperationCostsAllFolds_exactRecall))
    # allFeatureCostsstr.append(evaluation.getDetailedStr(testFeatureCostsAllFolds))
    # allRecallstr.append(evaluation.getDetailedStr(testRecallAllFolds_exactRecall))
    
    resultsRecorder.addAll(falsePositiveCost, (testTotalCostsAllFolds, testFeatureCostsAllFolds, testMisClassificationCostsAllFolds, testAccuracyAllFolds, testAUCAllFolds, testRecallAllFolds, testFDRAllFolds, testOperationCostsAllFolds, testRecallAllFolds_exactRecall, testFDRAllFolds_exactRecall, testOperationCostsAllFolds_exactRecall))

if COST_TYPE == "recall":
    resultsRecorder.writeOutResults(dataName + "_Shim2018_" + str(targetRecall) + COST_TYPE)
else:
    resultsRecorder.writeOutResults(dataName + "_Shim2018_" + COST_TYPE)

print("FINISHED ALL")

# allFalsePositiveCosts_asString = [str(cost) for cost in constants.allFalsePositiveCosts]
# 
# print("")
# print("LATEX SUMMARY: ")
# print(" & " + " & ".join(allFalsePositiveCosts_asString)  + " \\\\")
# print("FDR & " + " & ".join(allFDRstr))
# print("OPC & " + " & ".join(allOPCstr))
# print("Costs of acquired features & " + " & ".join(allFeatureCostsstr))
# print("Recall & " + " & ".join(allRecallstr))

import numpy
import experimentSettingBaselines
import experimentHelper
import evaluation
import os

import constants

# dataName = "pima_5foldCV"
dataName = "breastcancer_5foldCV"
# dataName = "pyhsioNetWithMissing_5foldCV"
# dataName = "heartDiseaseWithMissing_5foldCV"

COST_TYPE = "asymmetricCost"
# COST_TYPE = "symmetricCost"


classificationModelName = "Combined"
# classificationModelName = "logReg"


if COST_TYPE == "symmetricCost":
    # only used to compare with the results from previous work on Diabetes data
    assert(dataName == "pima_5foldCV")
    ALL_FALSE_POSITIVE_COSTS = [400,800]
    correctClassificationCost = -50.0 # in order to align to the setting in (Ji and Carin, 2007; Dulac-Arnold et al., 2012) for Diabetes.
    
elif COST_TYPE == "asymmetricCost":
    ALL_FALSE_POSITIVE_COSTS = constants.allFalsePositiveCosts
else:
    assert(False)


resultsRecorder = experimentHelper.ResultsRecorder(len(ALL_FALSE_POSITIVE_COSTS))
readyForSaving = False


for falsePositiveCost in ALL_FALSE_POSITIVE_COSTS:
    
    for testFoldId in range(constants.NUMBER_OF_FOLDS):
     
        print("*************************** ADAPT GBRT *******************************************")
        
        if COST_TYPE == "asymmetricCost":
            falseNegativeCost = falsePositiveCost * constants.FN_TO_FP_RATIO
            bestIdOnValid = evaluation.getBestAverage10TrainFoldTotalCostsResultAsymmetricForADAPTGBRT(classificationModelName, dataName, testFoldId, falsePositiveCost, falseNegativeCost)
        elif COST_TYPE == "symmetricCost":
            print("****************************")
            print("testFoldId = ", testFoldId)
            bestIdOnValid = evaluation.getBestAverage10TrainFoldTotalCostsResultSymmetricForADAPTGBRT(classificationModelName, dataName, testFoldId, symmetricMisclassificationCost = falsePositiveCost, sameClassCost = correctClassificationCost)
        
            
        evaluation.saveBestHyperparameterStringForADAPTGBRT(bestIdOnValid, classificationModelName, dataName, testFoldId, falsePositiveCost, COST_TYPE)
         
     
    finalResultsFilename = experimentSettingBaselines.MATLAB_FOLDER_RESULTS_ADAPT_GBRT + dataName + "_" + str(int(falsePositiveCost)) + '_forFinalTrainingAndTesting_' + str(constants.NUMBER_OF_FOLDS-1) + "_" + COST_TYPE + "_" + classificationModelName + "_allResults.csv"
    if os.path.isfile(finalResultsFilename):
        
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
        
            finalResultsFilename = experimentSettingBaselines.MATLAB_FOLDER_RESULTS_ADAPT_GBRT + dataName + "_" + str(int(falsePositiveCost)) + '_forFinalTrainingAndTesting_' + str(testFoldId) + "_" + COST_TYPE + "_" + classificationModelName + "_allResults.csv" 
            allResults = numpy.loadtxt(finalResultsFilename, delimiter = ",")
            assert(len(allResults.shape) == 1 and allResults.shape[0] == 4)
            
            testAccuracy = allResults[0]
            testFeatureCosts = allResults[1]
            
            if COST_TYPE == "asymmetricCost":
                falseNegativeCost = falsePositiveCost * constants.FN_TO_FP_RATIO
                testTotalCosts = evaluation.getAverageTotalCosts_forADAPTGBRT(allResults[1], allResults[2], allResults[3], falsePositiveCost, falseNegativeCost)
            elif COST_TYPE == "symmetricCost":
                testTotalCosts = evaluation.getTotalCostsSimple(allResults[0], allResults[1], misclassificationCostSymmetric = falsePositiveCost, sameClassCost = correctClassificationCost)
                
            testTotalCostsAllFolds[testFoldId] = testTotalCosts
            testFeatureCostsAllFolds[testFoldId] = testFeatureCosts
            testMisClassificationCostsAllFolds[testFoldId] = testTotalCosts - testFeatureCosts  
            testAccuracyAllFolds[testFoldId] = testAccuracy
         
        resultsRecorder.addAll(falsePositiveCost, (testTotalCostsAllFolds, testFeatureCostsAllFolds, testMisClassificationCostsAllFolds, testAccuracyAllFolds, testAUCAllFolds, testRecallAllFolds, testFDRAllFolds, testOperationCostsAllFolds, testRecallAllFolds_exactRecall, testFDRAllFolds_exactRecall, testOperationCostsAllFolds_exactRecall))
        readyForSaving = True

if readyForSaving:
    filename = dataName + "_AdaptGBRT_" + classificationModelName + "_" + COST_TYPE
    resultsRecorder.writeOutResults(filename)
    print("WROTE OUT FINAL RESULTS INTO: " + filename)
    
print("Finished Successfully")
    



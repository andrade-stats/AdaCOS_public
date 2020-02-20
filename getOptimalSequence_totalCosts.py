import numpy
import realdata
import sklearn.metrics
import evaluation
import experimentHelper
import dynamicAcquisition
import time
import sys
import prepareFeatureSets
from multiprocessing import Pool
import preprocessing
import constants
import pickle
from pathlib import Path


# /opt/intel/intelpython3/bin/python getOptimalSequence_mixed.py breastcancer_5foldCV Combined asymmetricCost nonLinearL1
# /opt/intel/intelpython3/bin/python getOptimalSequence_mixed.py pima_5foldCV Combined asymmetricCost
# /opt/intel/intelpython3/bin/python getOptimalSequence_mixed.py heartDiseaseWithMissing_5foldCV Combined asymmetricCost
# /opt/intel/intelpython3/bin/python getOptimalSequence_mixed.py pyhsioNetWithMissing_5foldCV Combined asymmetricCost greedy

# /opt/intel/intelpython3/bin/python getOptimalSequence_mixed.py pima_5foldCV Combined symmetricCost greedy
# /opt/intel/intelpython3/bin/python getOptimalSequence_mixed.py pima_5foldCV Combined symmetricCost nonLinearL1

dataName = sys.argv[1]

USE_UNLABELED_DATA = False
# assert(sys.argv[2] == "USE_UNLABELED_DATA" or sys.argv[2] == "NO_UNLABELED_DATA")
# USE_UNLABELED_DATA = (sys.argv[2] == "USE_UNLABELED_DATA")


onlyLookingOneStepAhead = False
NR_OF_USED_CPUS = None
NUMBER_OF_SAMPLES = None
    
classificationModelName = sys.argv[2]
assert(classificationModelName == "logReg" or classificationModelName == "GAM" or classificationModelName == "Combined")

COST_TYPE = sys.argv[3] 


densityRegressionModelName = "BR"
# densityRegressionModelName = "OrdinaryRegression"

assert(dataName.endswith("5foldCV"))

assert(sys.argv[4] == "l1" or sys.argv[4] == "greedy" or sys.argv[4] == "nonLinearL1"  or sys.argv[4] == "mixed")
FEATURE_SELECTION_METHOD = sys.argv[4]


if COST_TYPE == "symmetricCost":
    # only used to compare with the results from previous work on Diabetes data
    assert(dataName == "pima_5foldCV")
    ALL_FALSE_POSITIVE_COSTS = [400,800]
elif COST_TYPE == "asymmetricCost":
    ALL_FALSE_POSITIVE_COSTS = constants.allFalsePositiveCosts
else:
    assert(False)
    

DYNAMIC = "dynamic"
STATIC = "static"
FULL_MODEL = "fullModel"
allVariations = [FULL_MODEL, DYNAMIC, STATIC]





startTimeTotal = time.time()


definedFeatureCosts = realdata.getFeaturesCosts(dataName)


trainedModelsFilenameNonLinearL1 = dataName + "_" + classificationModelName + "_nonLinearL1"

with open(constants.MODEL_FOLDERNAME + trainedModelsFilenameNonLinearL1 + "_models", "rb") as f:
    allPredictionModelsNonLinearL1_allFolds = pickle.load(f)
with open(constants.MODEL_FOLDERNAME + trainedModelsFilenameNonLinearL1 + "_probs", "rb") as f:
    allTrainingTrueProbsAllModelsNonLinearL1_allFolds = pickle.load(f)
with open(constants.MODEL_FOLDERNAME + trainedModelsFilenameNonLinearL1 + "_features", "rb") as f:
    allFeatureArraysInOrderNonLinearL1_allFolds = pickle.load(f)






for variationName in allVariations:
    
    infoStr = variationName
    
    if variationName == DYNAMIC:
        infoStr += "_" + densityRegressionModelName
        
        if USE_UNLABELED_DATA:
            infoStr += "_withUnlabeledData"
        else:
            infoStr += "_noUnlabeledData"
    
    
    if variationName == FULL_MODEL:
        infoStr = FULL_MODEL
    else:
        infoStr += "_" + FEATURE_SELECTION_METHOD
        
    infoStr += "_" + classificationModelName
    
    
    resultsRecorder = experimentHelper.ResultsRecorder(len(ALL_FALSE_POSITIVE_COSTS))
    
    for costId in range(len(ALL_FALSE_POSITIVE_COSTS)):
        
        falsePositiveCost = ALL_FALSE_POSITIVE_COSTS[costId]
        
        if COST_TYPE == "symmetricCost":
            falseNegativeCost = falsePositiveCost
            assert(dataName == "pima_5foldCV")
            correctClassificationCost = -50.0 # in order to align to the setting in (Ji and Carin, 2007; Dulac-Arnold et al., 2012) for Diabetes.
            
            misclassificationCosts = numpy.zeros((2, 2))
            misclassificationCosts[0, 1] = falsePositiveCost 
            misclassificationCosts[1, 0] = falseNegativeCost 
            misclassificationCosts[0, 0] = correctClassificationCost
            misclassificationCosts[1, 1] = correctClassificationCost
        elif COST_TYPE == "asymmetricCost":
             
            falseNegativeCost = falsePositiveCost * constants.FN_TO_FP_RATIO
            
            misclassificationCosts = numpy.zeros((2, 2))
            misclassificationCosts[0, 1] = falsePositiveCost 
            misclassificationCosts[1, 0] = falseNegativeCost 
            misclassificationCosts[0, 0] = 0.0
            misclassificationCosts[1, 1] = 0.0
        else:
            assert(False)
        
        # print("falsePositiveCost = ", falsePositiveCost)
        # print("falseNegativeCost = ", falseNegativeCost)
            
        numpy.random.seed(3523421)
        
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
        
        
        if dataName != "pyhsioNetWithMissing_5foldCV":
            trainedModelsFilenameGreedy = dataName + "_" + classificationModelName + "_" + COST_TYPE + "_" + str(falsePositiveCost) + "_greedy"
    
            with open(constants.MODEL_FOLDERNAME + trainedModelsFilenameGreedy + "_models", "rb") as f:
                allPredictionModelsGreedy_allFolds = pickle.load(f)
            with open(constants.MODEL_FOLDERNAME + trainedModelsFilenameGreedy + "_probs", "rb") as f:
                allTrainingTrueProbsAllModelsGreedy_allFolds = pickle.load(f)
            with open(constants.MODEL_FOLDERNAME + trainedModelsFilenameGreedy + "_features", "rb") as f:
                allFeatureArraysInOrderGreedy_allFolds = pickle.load(f)
                    
        
        startTime = time.time()
        
        runTimesAllFolds = numpy.zeros(constants.NUMBER_OF_FOLDS)
        
        for foldId in range(constants.NUMBER_OF_FOLDS):
            
            # *******************************************************************************************************************
            # ********************************* get feature sets, prediction models, and predictedTrueProbs(on training data)  ******************************
            # *******************************************************************************************************************
        
            trainData, trainLabels, unlabeledData, testData, testLabels = realdata.loadSubset(dataName, None, foldId, constants.IMPUTATION_METHOD)
            
            if USE_UNLABELED_DATA:
                assert(unlabeledData.shape[0] > 0)
            else:
                unlabeledData = numpy.zeros((0, trainData.shape[1]))
            
            allData = numpy.vstack((trainData, unlabeledData))
            
            assert(definedFeatureCosts.shape[0] == allData.shape[1])
            
            print("GET ALL PREDICTION MODEL AND DETERMINE FALSE NEGATIVE COSTS: ")
            
            
            # ************************************************************
            # ****** decide on covariate acquisition path  *******
            # *********************************************************
            
            if FEATURE_SELECTION_METHOD == "mixed":
                allPredictionModels = allPredictionModelsNonLinearL1_allFolds[foldId]
                allTrainingTrueProbsAllModels = allTrainingTrueProbsAllModelsNonLinearL1_allFolds[foldId]
                allFeatureArraysInOrder = allFeatureArraysInOrderNonLinearL1_allFolds[foldId]
    
                if dataName != "pyhsioNetWithMissing_5foldCV": 
                    
                    bestTotalCostGreedy, bestModelIdGreedy = prepareFeatureSets.selectFeatureSets(definedFeatureCosts, misclassificationCosts, trainLabels, allFeatureArraysInOrderGreedy_allFolds[foldId], allTrainingTrueProbsAllModelsGreedy_allFolds[foldId])
                    bestTotalCostNonLinearL1, bestModelIdNonLinearL1 = prepareFeatureSets.selectFeatureSets(definedFeatureCosts, misclassificationCosts, trainLabels, allFeatureArraysInOrderNonLinearL1_allFolds[foldId], allTrainingTrueProbsAllModelsNonLinearL1_allFolds[foldId])
        
                    if bestTotalCostGreedy < bestTotalCostNonLinearL1:
                        allPredictionModels = allPredictionModelsGreedy_allFolds[foldId]
                        allTrainingTrueProbsAllModels = allTrainingTrueProbsAllModelsGreedy_allFolds[foldId]
                        allFeatureArraysInOrder = allFeatureArraysInOrderGreedy_allFolds[foldId]
            elif FEATURE_SELECTION_METHOD == "nonLinearL1":
                allPredictionModels = allPredictionModelsNonLinearL1_allFolds[foldId]
                allTrainingTrueProbsAllModels = allTrainingTrueProbsAllModelsNonLinearL1_allFolds[foldId]
                allFeatureArraysInOrder = allFeatureArraysInOrderNonLinearL1_allFolds[foldId]
            elif FEATURE_SELECTION_METHOD == "greedy":
                allPredictionModels = allPredictionModelsGreedy_allFolds[foldId]
                allTrainingTrueProbsAllModels = allTrainingTrueProbsAllModelsGreedy_allFolds[foldId]
                allFeatureArraysInOrder = allFeatureArraysInOrderGreedy_allFolds[foldId]
            else:
                assert(False)

            # **************************************************
            # ****** start evaluation on test data *******
            # **************************************************
            
            if variationName == STATIC or variationName == FULL_MODEL:
                if variationName == STATIC:
                    print("USE STATIC SELECTION ")
                    _, bestModelId = prepareFeatureSets.selectFeatureSets(definedFeatureCosts, misclassificationCosts, trainLabels, allFeatureArraysInOrder, allTrainingTrueProbsAllModels)
                else:
                    print("USE FULL MODEL ")
                    assert(variationName == FULL_MODEL)
                    bestModelId = len(allFeatureArraysInOrder) - 1
                    assert(len(allFeatureArraysInOrder[bestModelId]) == trainData.shape[1])
                    
                bestPredictionModel = allPredictionModels[bestModelId]
                selectedFeatureIds = allFeatureArraysInOrder[bestModelId]
                
                observedCovariatesForClassifier = testData[:,selectedFeatureIds] 
                predictedProbs = bestPredictionModel.predict_proba(observedCovariatesForClassifier)
                predictedProbs = evaluation.ensure2D(predictedProbs)
                predictedTestLabels = evaluation.bayesClassifier(predictedProbs, misclassificationCosts)
                
                predictedTestTrueLabelProbs = predictedProbs[:, 1]
                avgTestFeatureCosts = numpy.sum(definedFeatureCosts[selectedFeatureIds])
                
            else:
                
                    # *******************************************************************************************************************
                    # ********************************* prepare and run dynamic acquisition ******************************
                    # *******************************************************************************************************************
               
                allMisclassificationCosts = []
                for i in range(len(allFeatureArraysInOrder)):
                    allMisclassificationCosts.append(misclassificationCosts)
                
                print("PREPARE REGRESSION MODELS: ")
                if densityRegressionModelName == "OrdinaryRegression":
                    assert(False)
                    allSamplerInfos = dynamicAcquisition.prepareSamplerAndClassifierForTest(allData, allFeatureArraysInOrder)
                else:
                    allSamplerInfos = dynamicAcquisition.prepareSamplerAndClassifierForTestWithRegression(allPredictionModels, allData, allFeatureArraysInOrder, densityRegressionModelName)
            
                 
                print("START DYNAMIC SEARCH: ")
                startTime = time.time()
                       
                allParamsForMultiprocessMap = []
                for i in range(testData.shape[0]):
                    testSample = testData[i]
                    allParamsForMultiprocessMap.append((i, definedFeatureCosts, allFeatureArraysInOrder, allSamplerInfos, allPredictionModels, allMisclassificationCosts, testSample, onlyLookingOneStepAhead, NUMBER_OF_SAMPLES))
                 
                with Pool(NR_OF_USED_CPUS) as pool:
                    allResults = pool.starmap(dynamicAcquisition.runDynamicAcquisition_bayesRisk, allParamsForMultiprocessMap)
                  
                assert(len(allResults) == testData.shape[0])
                predictedTestLabels = numpy.zeros(testData.shape[0], dtype=numpy.int)
                predictedTestTrueLabelProbs = numpy.zeros(testData.shape[0])
                totalTestFeatureCosts = 0.0 
                for i in range(testData.shape[0]):
                    queriedFeatures, acquiredFeaturesCost, predictedLabel, predictedTrueLabelProb = allResults[i]
                    print(str(queriedFeatures) + " | " + str(acquiredFeaturesCost) + " | " + str(predictedLabel) + " | " + str(predictedTrueLabelProb))
                    predictedTestLabels[i] = predictedLabel
                    predictedTestTrueLabelProbs[i] = predictedTrueLabelProb
                    totalTestFeatureCosts += acquiredFeaturesCost
                 
                runTime = (time.time() - startTime) / float(testData.shape[0])
                print("runtime per test sample (in seconds) = " + str(runTime))
                runTimesAllFolds[foldId] = runTime
                 
                avgTestFeatureCosts = totalTestFeatureCosts / float(testData.shape[0])
              
            
            threshold_forExactRecall = evaluation.getThresholdFromPredictedProbabilities(testLabels, predictedTestTrueLabelProbs, targetRecall = 0.95)
            testRecallAllFolds_exactRecall[foldId] = evaluation.getRecall(testLabels, predictedTestTrueLabelProbs, threshold_forExactRecall)
            testFDRAllFolds_exactRecall[foldId] = evaluation.getFDR(testLabels, predictedTestTrueLabelProbs, threshold_forExactRecall)
            testTotalCostsAllFolds[foldId] = evaluation.getAverageTotalCosts(testLabels, predictedTestLabels, avgTestFeatureCosts, misclassificationCosts)
            testMisClassificationCostsAllFolds[foldId] = evaluation.getAverageMisclassificationCosts(testLabels, predictedTestLabels, misclassificationCosts)
        
            testFeatureCostsAllFolds[foldId] = avgTestFeatureCosts
            
            testAccuracyAllFolds[foldId] = sklearn.metrics.accuracy_score(testLabels, predictedTestLabels)
            testAUCAllFolds[foldId] = sklearn.metrics.roc_auc_score(testLabels, predictedTestTrueLabelProbs)
        
            testRecallAllFolds[foldId] = sklearn.metrics.recall_score(testLabels, predictedTestLabels)
            testFDRAllFolds[foldId] = 1.0 - sklearn.metrics.precision_score(testLabels, predictedTestLabels)
            testOperationCostsAllFolds[foldId] = evaluation.getAverageOperationCosts(testLabels, predictedTestLabels, avgTestFeatureCosts, falsePositiveCost)
            
            
        print("NEW ULTRA FAST VERSION")
        print("dataName = ", dataName)
        print("NUMBER_OF_FOLDS = ", constants.NUMBER_OF_FOLDS)
        print("onlyLookingOneStepAhead = ", onlyLookingOneStepAhead)
        print("densityRegressionModelName = ", densityRegressionModelName)
        print("falseNegativeCost = ", falseNegativeCost)
        
        print("RESULTS WITH DYNAMIC ACQUISTION (SLOW VERSION): ")
        print("*************************** AVERAGE OVER ALL FOLDS *******************************************")
        evaluation.showHelper("total costs = ", testTotalCostsAllFolds)
        evaluation.showHelper("feature costs = ", testFeatureCostsAllFolds)
        evaluation.showHelper("misclassification costs = ", testMisClassificationCostsAllFolds)
        evaluation.showHelper("accuracy = ", testAccuracyAllFolds)
        evaluation.showHelper("AUC = ", testAUCAllFolds)
        resultsRecorder.addAll(falsePositiveCost, (testTotalCostsAllFolds, testFeatureCostsAllFolds, testMisClassificationCostsAllFolds, testAccuracyAllFolds, testAUCAllFolds, testRecallAllFolds, testFDRAllFolds, testOperationCostsAllFolds, testRecallAllFolds_exactRecall, testFDRAllFolds_exactRecall, testOperationCostsAllFolds_exactRecall))
        
    print("variationName = ", variationName)
    print("classificationModelName = ", classificationModelName)
    print("densityRegressionModelName = ", densityRegressionModelName)
    print("ALL_FALSE_POSITIVE_COSTS = ", ALL_FALSE_POSITIVE_COSTS)
    print("total runtime (in minutes) = " + str((time.time() - startTimeTotal) / 60.0))
    
    resultsRecorder.writeOutResults(dataName + "_getOptimalSequence_" + infoStr + "_" + COST_TYPE)
    
    print("FEATURE_SELECTION_METHOD = ", FEATURE_SELECTION_METHOD)
    print("Finished Successfully")


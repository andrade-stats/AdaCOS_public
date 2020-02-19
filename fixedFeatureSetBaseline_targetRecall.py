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
from multiprocessing import Pool
import sys
import constants

# /opt/intel/intelpython3/bin/python fixedFeatureSetBaseline_targetRecall.py breastcancer_5foldCV NO_UNLABELED_DATA fullModel logReg
# /opt/intel/intelpython3/bin/python fixedFeatureSetBaseline_targetRecall.py pyhsioNetWithMissing_5foldCV NO_UNLABELED_DATA fullModel logReg
# /opt/intel/intelpython3/bin/python fixedFeatureSetBaseline_targetRecall.py pima_5foldCV NO_UNLABELED_DATA fullModel logReg
# /opt/intel/intelpython3/bin/python fixedFeatureSetBaseline_targetRecall.py heartDiseaseWithMissing_5foldCV NO_UNLABELED_DATA fullModel logReg

     
dataName = sys.argv[1]

assert(sys.argv[2] == "USE_UNLABELED_DATA" or sys.argv[2] == "NO_UNLABELED_DATA")
USE_UNLABELED_DATA = (sys.argv[2] == "USE_UNLABELED_DATA")


if USE_UNLABELED_DATA:
    infoStr = "withUnlabeledData"
else:
    infoStr = "noUnlabeledData"

assert(sys.argv[3] == "l1" or sys.argv[3] == "greedy" or sys.argv[3] == "fullModel")
USE_L1 = (sys.argv[3] == "l1") 

if USE_L1:
    infoStr += "_l1"

FULL_MODEL = (sys.argv[3] == "fullModel")


classificationModelName = sys.argv[4]
assert(classificationModelName == "logReg" or classificationModelName == "GAM")

targetRecall = float(sys.argv[5])
assert(targetRecall == 0.95 or targetRecall == 0.99)


infoStr += "_" + str(targetRecall) + "targetRecall"

imputationMethod = "gaussian_imputation"

NUMBER_OF_FOLDS = 5

startTime = time.time()


resultsRecorder = experimentHelper.ResultsRecorder(len(constants.allFalsePositiveCosts))

for falsePositiveCost in constants.allFalsePositiveCosts:
    
    numpy.random.seed(3523421)

    def evalOneFold(foldId):
        
        definedFeatureCosts = realdata.getFeaturesCosts(dataName)
            
        trainData, trainLabels, unlabeledData, testData, testLabels = realdata.loadSubset(dataName, None, foldId, imputationMethod)
        
        if USE_UNLABELED_DATA:
            assert(unlabeledData.shape[0] > 0)
        else:
            unlabeledData = numpy.zeros((0, trainData.shape[1]))
        
        allData = numpy.vstack((trainData, unlabeledData))
        
        assert(definedFeatureCosts.shape[0] == allData.shape[1])
        
        print("training data size = ", trainData.shape[0])
        print("unlabeled data size = ", unlabeledData.shape[0])
        print("test data size = ", testData.shape[0])
        
        print("*****************************")
        print("foldId = ", foldId)
        print("*****************************")
        
        if FULL_MODEL:
            bestFixedFeatures = numpy.arange(trainData.shape[1])
            # print("bestFixedFeatures = ", bestFixedFeatures)
            # assert(False)
            bestModel, misclassificationCosts, totalCostEstimate = prepareFeatureSets.getPredictionModelsAndCosts(trainData, trainLabels, bestFixedFeatures, definedFeatureCosts, falsePositiveCost, targetRecall, useTargetRecall = True, falseNegativeCost = None, classificationModelName = classificationModelName)
            
        else:
            if USE_L1:
                allFeatureSetsInOrder, _ = prepareFeatureSets.getAllFeatureSetsInOrderWithL1LogReg(trainData, trainLabels, unlabeledData, None, definedFeatureCosts)
            else:
                print("NOT YET SUPPORTED !!")
                assert(False)
                # allFeatureSetsInOrder, allEstimatedTotalCosts = prepareFeatureSets.getAllFeatureSetsInOrderWithGreedyMethod(trainData, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts)
            
        
            print("GET ALL PREDICTION MODEL AND DETERMINE FALSE NEGATIVE COSTS: ")  
            allPredictionModels, allMisclassificationCosts, allEstimatedTotalCosts = prepareFeatureSets.getAllPredictionModelsAndCosts(trainData, trainLabels, allFeatureSetsInOrder, definedFeatureCosts, falsePositiveCost, targetRecall, useTargetRecall = True, falseNegativeCost = None, classificationModelName = classificationModelName)
            
            bestModelId = numpy.argmin(allEstimatedTotalCosts)
            bestModel = allPredictionModels[bestModelId]
            misclassificationCosts = allMisclassificationCosts[bestModelId]
            bestFixedFeatures = allFeatureSetsInOrder[bestModelId]
       
        return evaluation.getOverallPerformance_fixedCovariateSet(bestModel, testData, testLabels, definedFeatureCosts, misclassificationCosts, bestFixedFeatures, targetRecall)
    
    allParamsForMultiprocessMap = numpy.arange(NUMBER_OF_FOLDS)
      
    with Pool() as pool:
        allResults = pool.map(evalOneFold, allParamsForMultiprocessMap)
    
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
      
    for foldId in range(NUMBER_OF_FOLDS):
        testTotalCostsAllFolds[foldId], testFeatureCostsAllFolds[foldId], testMisClassificationCostsAllFolds[foldId], testAccuracyAllFolds[foldId], testAUCAllFolds[foldId], testRecallAllFolds[foldId], testFDRAllFolds[foldId], testOperationCostsAllFolds[foldId], testRecallAllFolds_exactRecall[foldId], testFDRAllFolds_exactRecall[foldId], testOperationCostsAllFolds_exactRecall[foldId]  = allResults[foldId]
        
    print("test runtime (in minutes) = " +  str((time.time() - startTime) / 60.0))
    
    print("USE_UNLABELED_DATA = ", USE_UNLABELED_DATA)
    print("dataName = ", dataName)
    print("NUMBER_OF_FOLDS = ", NUMBER_OF_FOLDS)
    print("FULL_MODEL = ", FULL_MODEL)
    print(infoStr)
    
    print("*************************** AVERAGE OVER ALL FOLDS *******************************************")
    evaluation.showHelper("total costs = ", testTotalCostsAllFolds)
    evaluation.showHelper("feature costs = ", testFeatureCostsAllFolds)
    evaluation.showHelper("misclassification costs = ", testMisClassificationCostsAllFolds)
    evaluation.showHelper("accuracy = ", testAccuracyAllFolds)
    evaluation.showHelper("AUC = ", testAUCAllFolds)
    resultsRecorder.addAll(falsePositiveCost, (testTotalCostsAllFolds, testFeatureCostsAllFolds, testMisClassificationCostsAllFolds, testAccuracyAllFolds, testAUCAllFolds, testRecallAllFolds, testFDRAllFolds, testOperationCostsAllFolds, testRecallAllFolds_exactRecall, testFDRAllFolds_exactRecall, testOperationCostsAllFolds_exactRecall))

if FULL_MODEL:
    resultsRecorder.writeOutResults(dataName + "_fullModel_" + str(targetRecall) + "targetRecall")
else:
    resultsRecorder.writeOutResults(dataName + "_fixedFeatureSetBaseline_" + infoStr)

print("Finished Successfully")



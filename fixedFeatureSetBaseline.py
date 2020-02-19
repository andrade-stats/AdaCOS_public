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


# currently running:
# manto52,53,54
# /opt/intel/intelpython3/bin/python fixedFeatureSetBaseline.py miniBooNE_5foldCV NO_UNLABELED_DATA
# /opt/intel/intelpython3/bin/python fixedFeatureSetBaseline.py miniBooNE_5foldCV USE_UNLABELED_DATA
# /opt/intel/intelpython3/bin/python fixedFeatureSetBaseline.py breastcancer_5foldCV NO_UNLABELED_DATA greedy
# /opt/intel/intelpython3/bin/python fixedFeatureSetBaseline.py pima_5foldCV NO_UNLABELED_DATA

# /opt/intel/intelpython3/bin/python fixedFeatureSetBaseline.py pyhsioNetNoMissing_5foldCV NO_UNLABELED_DATA greedy

# dataName = "pima_5foldCV"
# dataName = "breastcancer_5foldCV"
# dataName = "miniBooNE_5foldCV"

# /opt/intel/intelpython3/bin/python fixedFeatureSetBaseline.py breastcancer_5foldCV NO_UNLABELED_DATA l1

     
dataName = sys.argv[1]


if dataName == "pima_5foldCV":
    sameClassCost = -50.0 # set to -50.0 in order to compare to Li and Carin work
    # assert(misclassificationCostsSymmetric == 400 or misclassificationCostsSymmetric == 800)
else:
    sameClassCost = 0.0


assert(sys.argv[2] == "USE_UNLABELED_DATA" or sys.argv[2] == "NO_UNLABELED_DATA")
USE_UNLABELED_DATA = (sys.argv[2] == "USE_UNLABELED_DATA")

assert(sys.argv[3] == "l1" or sys.argv[3] == "greedy")
USE_L1 = (sys.argv[3] == "l1") 

SMALL_COSTS = False

if USE_UNLABELED_DATA:
    infoStr = "withUnlabeledData"
else:
    infoStr = "noUnlabeledData"

if USE_L1:
    infoStr += "_l1"


if SMALL_COSTS:
    infoStr += "_smallCosts"
    allMisclassificationCostsSymmetric = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
else:
    allMisclassificationCostsSymmetric = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    

# ONLY FOR DEBUGGING
# DEBUG_MODE = True
# allMisclassificationCostsSymmetric = [100000.0] #  [800.0]
# NUMBER_OF_FOLDS = 5

NUMBER_OF_FOLDS = 5

startTime = time.time()


resultsRecorder = experimentHelper.ResultsRecorder(len(allMisclassificationCostsSymmetric))

for misclassificationCostsSymmetric in allMisclassificationCostsSymmetric:
    
    numpy.random.seed(3523421)

    def evalOneFold(foldId):
        
        # row id = true class id, column id = predicted class id
        misclassificationCosts = numpy.zeros((2,2))
        misclassificationCosts[0, 1] = misclassificationCostsSymmetric 
        misclassificationCosts[1, 0] = misclassificationCostsSymmetric
        misclassificationCosts[0, 0] = sameClassCost
        misclassificationCosts[1, 1] = sameClassCost
    
        definedFeatureCosts = realdata.getFeaturesCosts(dataName)
    
        trainData, trainLabels, unlabeledData, testData, testLabels = realdata.loadSubsetNew(dataName, None, foldId)
        trainLabels = experimentHelper.getLabelsStartingAtZero(trainLabels)
        testLabels = experimentHelper.getLabelsStartingAtZero(testLabels)
    
        if USE_UNLABELED_DATA:
            assert(unlabeledData.shape[0] > 0)
        else:
            unlabeledData = numpy.zeros((0,trainData.shape[1]))
            
        print("training data size = ", trainData.shape[0])
        print("unlabeled data size = ", unlabeledData.shape[0])
        print("test data size = ", testData.shape[0])
        
        print("*****************************")
        print("foldId = ", foldId)       
        print("*****************************")
        
        if USE_L1:
            allFeatureSetsInOrder, allEstimatedTotalCosts = prepareFeatureSets.getAllFeatureSetsInOrderWithL1LogReg(trainData, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts)
        else:
            allFeatureSetsInOrder, allEstimatedTotalCosts = prepareFeatureSets.getAllFeatureSetsInOrderWithGreedyMethod(trainData, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts)
            
            # print("number of variabels = ", trainData.shape[1])
            # print("allFeatureSetsInOrder size = ", len(allFeatureSetsInOrder))
            # assert(False)
            
        bestFixedFeatures = allFeatureSetsInOrder[numpy.argmin(allEstimatedTotalCosts)]
        
        bestModel, _ = prepareFeatureSets.getOptimalTrainedModel_withRecallThresholds(trainData, trainLabels, bestFixedFeatures, 0.5)
        
        return evaluation.getOverallPerformance_fixedCovariateSet(bestModel, testData, testLabels, definedFeatureCosts, misclassificationCosts, bestFixedFeatures, targetRecall = 0.95)
        
        # bestLogRegModel, _ = prepareFeatureSets.getOptimalTrainedModel(trainData, trainLabels, bestFixedFeatureIds)
        # testTotalCostsThisFold, testFeatureCostsThisFold, testMisClassificationCostsThisFold, testAccuracyThisFold = evaluation.getTestDataAverageTotalCostsWithFixedFeatureIds(bestLogRegModel, testData, testLabels, misclassificationCosts, bestFixedFeatureIds, definedFeatureCosts)
        # testAUCThisFold = evaluation.getTestAUC(bestLogRegModel, testData[:, bestFixedFeatureIds], testLabels)
        # print("bestFeatureSet = ", bestFixedFeatureIds)
        # print("bestCosts = ", numpy.min(allEstimatedTotalCosts))
        # print("testAccuracyThisFold = ", testAccuracyThisFold)
        # print("testAUCThisFold = ", testAUCThisFold)
        # return testTotalCostsThisFold, testFeatureCostsThisFold, testMisClassificationCostsThisFold, testAccuracyThisFold, testAUCThisFold
        
    
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
    print(infoStr)
    
    print("*************************** AVERAGE OVER ALL FOLDS (misclassification costs = " + str(misclassificationCostsSymmetric) + ", sameClassCost = " + str(sameClassCost) + ") *******************************************")
    evaluation.showHelper("total costs = ", testTotalCostsAllFolds)
    evaluation.showHelper("feature costs = ", testFeatureCostsAllFolds)
    evaluation.showHelper("misclassification costs = ", testMisClassificationCostsAllFolds)
    evaluation.showHelper("accuracy = ", testAccuracyAllFolds)
    evaluation.showHelper("AUC = ", testAUCAllFolds)
    resultsRecorder.addAll(misclassificationCostsSymmetric, (testTotalCostsAllFolds, testFeatureCostsAllFolds, testMisClassificationCostsAllFolds, testAccuracyAllFolds, testAUCAllFolds, testRecallAllFolds, testFDRAllFolds, testOperationCostsAllFolds, testRecallAllFolds_exactRecall, testFDRAllFolds_exactRecall, testOperationCostsAllFolds_exactRecall))
    
resultsRecorder.writeOutResults(dataName + "_fixedFeatureSetBaseline_" + infoStr)

print("Finished Successfully")



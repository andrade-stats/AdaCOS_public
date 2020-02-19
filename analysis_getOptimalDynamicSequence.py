import numpy
import realdata
import sklearn.metrics
import evaluation
import experimentHelper
import fullDynamicAcquisition
import time
import sys
import prepareFeatureSets
from multiprocessing import Pool
from TrainedModelContainer import TrainedModelContainer
import multiprocessing

# currently running:

# manto51:
# /opt/intel/intelpython3/bin/python getOptimalDynamicSequence.py miniBooNE_5foldCV USE_UNLABELED_DATA

# /opt/intel/intelpython3/bin/python getOptimalDynamicSequence.py miniBooNE_5foldCV USE_UNLABELED_DATA
# /opt/intel/intelpython3/bin/python getOptimalDynamicSequence.py miniBooNE_5foldCV NO_UNLABELED_DATA
# /opt/intel/intelpython3/bin/python getOptimalDynamicSequence.py pima_5foldCV NO_UNLABELED_DATA

# /opt/intel/intelpython3/bin/python analysis_getOptimalDynamicSequence.py breastcancer_5foldCV NO_UNLABELED_DATA FullDynamicOrdinaryRegressionIMPROVED
# /opt/intel/intelpython3/bin/python analysis_getOptimalDynamicSequence.py breastcancer_5foldCV NO_UNLABELED_DATA FullDynamicOrdinaryRegressionSLOW
# /opt/intel/intelpython3/bin/python analysis_getOptimalDynamicSequence.py breastcancer_5foldCV NO_UNLABELED_DATA LinearDynamicOrdinaryRegression

dataName = sys.argv[1]

assert(sys.argv[2] == "USE_UNLABELED_DATA" or sys.argv[2] == "NO_UNLABELED_DATA")
USE_UNLABELED_DATA = (sys.argv[2] == "USE_UNLABELED_DATA")



onlyLookingOneStepAhead = False

# densityRegressionModelName = "FullDynamicOrdinaryRegressionSLOW"
# densityRegressionModelName = "LinearDynamicOrdinaryRegression"

densityRegressionModelName = sys.argv[3]


assert(dataName.endswith("5foldCV"))
    
if dataName == "pima_5foldCV":
    sameClassCost = -50.0  # set to -50.0 in order to compare to Li and Carin work
    # assert(misclassificationCostsSymmetric == 400 or misclassificationCostsSymmetric == 800)
else:
    sameClassCost = 0.0


SMALL_COSTS = False

infoStr = densityRegressionModelName

if USE_UNLABELED_DATA:
    infoStr += "_withUnlabeledData"
else:
    infoStr += "_noUnlabeledData"


if onlyLookingOneStepAhead:
    infoStr += "_onlyLookingOneStepAhead"

if SMALL_COSTS:
    infoStr += "_smallCosts"
    allMisclassificationCostsSymmetric = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
else:
    allMisclassificationCostsSymmetric = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    
NUMBER_OF_FOLDS = 5


NR_OF_USED_CPUS = None
NUMBER_OF_SAMPLES = None

runTimesAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
resultsRecorder = experimentHelper.ResultsRecorder(len(allMisclassificationCostsSymmetric))

startTimeTotal = time.time()

print("densityRegressionModelName = ", densityRegressionModelName)

misclassificationCostsSymmetric = 800
    
numpy.random.seed(3523421)

# row id = true class id, column id = predicted class id
misclassificationCosts = numpy.zeros((2, 2))
misclassificationCosts[0, 1] = misclassificationCostsSymmetric 
misclassificationCosts[1, 0] = misclassificationCostsSymmetric
misclassificationCosts[0, 0] = sameClassCost
misclassificationCosts[1, 1] = sameClassCost

testTotalCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
testFeatureCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
testMisClassificationCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
testAccuracyAllFolds = numpy.zeros(NUMBER_OF_FOLDS)

definedFeatureCosts = realdata.getFeaturesCosts(dataName)

startTime = time.time()

for foldId in range(NUMBER_OF_FOLDS):
    
    print("foldId = ", foldId)
    
    # *******************************************************************************************************************
    # ********************************* get feature sets  ******************************
    # *******************************************************************************************************************

    trainData, trainLabels, unlabeledData, testData, testLabels = realdata.loadSubsetNew(dataName, None, foldId)
    trainLabels = experimentHelper.getLabelsStartingAtZero(trainLabels)
    testLabels = experimentHelper.getLabelsStartingAtZero(testLabels)
    
    if USE_UNLABELED_DATA:
        assert(unlabeledData.shape[0] > 0)
    else:
        unlabeledData = numpy.zeros((0, trainData.shape[1]))
    
    allData = numpy.vstack((trainData, unlabeledData))
    
    
    # *******************************************************************************************************************
    # ********************************* prepare and run dynamic acquisition ******************************
    # *******************************************************************************************************************
    
    # testData = testData[0:40]
    
    if densityRegressionModelName == "LinearDynamicOrdinaryRegression":
        allFeatureSetsInOrder, _ = prepareFeatureSets.getAllFeatureSetsInOrderWithGreedyMethod(trainData, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts)
          
        fullCovMatrix, allSamplerInfos, allPredictionModels = fullDynamicAcquisition.prepareForLinearSequence(trainData, trainLabels, allData, allFeatureSetsInOrder)
              
        startTime = time.time()
        
        
        
        allParamsForMultiprocessMap = []
        for i in range(testData.shape[0]):
            testSample = testData[i]
            allParamsForMultiprocessMap.append((definedFeatureCosts, misclassificationCosts, allFeatureSetsInOrder, allSamplerInfos, allPredictionModels, testSample, onlyLookingOneStepAhead, fullCovMatrix))
        
        print("START LINEAR DYNAMIC SEARCH: ")
              
        with Pool(NR_OF_USED_CPUS) as pool:
            allResults = pool.starmap(fullDynamicAcquisition.runLinearSequenceDynamicAcquisition, allParamsForMultiprocessMap)
      
       
     
    elif densityRegressionModelName == "FullDynamicOrdinaryRegressionSLOW":
        assert(not onlyLookingOneStepAhead)
        
        fullCovMatrix = numpy.cov(allData.transpose(), bias=True)
        
        startTime = time.time()
           
        allParamsForMultiprocessMap = []
        for i in range(testData.shape[0]):
            testSample = testData[i]
            allParamsForMultiprocessMap.append((fullCovMatrix, trainData, trainLabels, testSample, misclassificationCosts, definedFeatureCosts))
        
        print("START FULL DYNAMIC SEARCH: ")
        
        with Pool(NR_OF_USED_CPUS) as pool:
            allResults = pool.starmap(fullDynamicAcquisition.fullDynamicGreedyMethodForDebugging, allParamsForMultiprocessMap)
        
    else:
        assert(densityRegressionModelName == "FullDynamicOrdinaryRegressionIMPROVED")
        assert(not onlyLookingOneStepAhead)
        
        fullCovMatrix = numpy.cov(allData.transpose(), bias=True)
        
        startTime = time.time()
          
        allFeatureSetsInOrder, allEstimatedTotalCosts = prepareFeatureSets.getAllFeatureSetsInOrderWithGreedyMethod(trainData, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts)
        bestAPrioriFeatureSet = allFeatureSetsInOrder[numpy.argmin(allEstimatedTotalCosts)]
        print("bestAPrioriFeatureSet = ", bestAPrioriFeatureSet)
        assert(len(bestAPrioriFeatureSet) >= 1)
        initialFeatureId = bestAPrioriFeatureSet[0]
        
        print("initialFeatureId = ", initialFeatureId)
        assert(False)
        
        modelContainer = TrainedModelContainer(trainData, trainLabels)
        manager = multiprocessing.Manager()
        lock = manager.Lock()
          
        allParamsForMultiprocessMap = []
        for i in range(testData.shape[0]):
            testSample = testData[i]
            allParamsForMultiprocessMap.append((fullCovMatrix, trainData, trainLabels, testSample, misclassificationCosts, definedFeatureCosts, initialFeatureId, modelContainer, lock))
        
        print("START FULL DYNAMIC SEARCH: ")
        
        with Pool(NR_OF_USED_CPUS) as pool:
            allResults = pool.starmap(fullDynamicAcquisition.fullDynamicGreedyMethod, allParamsForMultiprocessMap)
         
     
     
       
    assert(len(allResults) == testData.shape[0])
    predictedTestLabels = numpy.zeros(testData.shape[0], dtype=numpy.int)
    totalTestFeatureCosts = 0.0 
    for i in range(testData.shape[0]):
        queriedFeatures, acquiredFeaturesCost, predictedLabel = allResults[i]
        print(str(queriedFeatures) + " | " + str(acquiredFeaturesCost) + " | " + str(predictedLabel))
        predictedTestLabels[i] = predictedLabel
        totalTestFeatureCosts += acquiredFeaturesCost
    
    runTime = (time.time() - startTime) / float(testData.shape[0])
    print("runtime per test sample (in seconds) = " + str(runTime))
    runTimesAllFolds[foldId] = runTime
    
    avgTestFeatureCosts = totalTestFeatureCosts / float(testData.shape[0])
      
    testTotalCostsAllFolds[foldId] = evaluation.getAverageTotalCosts(testLabels, predictedTestLabels, avgTestFeatureCosts, misclassificationCosts)
    testFeatureCostsAllFolds[foldId] = avgTestFeatureCosts
    testMisClassificationCostsAllFolds[foldId] = evaluation.getAverageMisclassificationCosts(testLabels, predictedTestLabels, misclassificationCosts) 
    testAccuracyAllFolds[foldId] = sklearn.metrics.accuracy_score(testLabels, predictedTestLabels)

print("NEW ULTRA FAST VERSION")
print("dataName = ", dataName)
print("NUMBER_OF_FOLDS = ", NUMBER_OF_FOLDS)
print("onlyLookingOneStepAhead = ", onlyLookingOneStepAhead)
print("misclassificationCostsSymmetric = ", misclassificationCostsSymmetric)
print("sameClassCost = ", sameClassCost)
print("densityRegressionModelName = ", densityRegressionModelName)

print("RESULTS WITH DYNAMIC ACQUISTION (SLOW VERSION): ")
# print("VERSION WITH GUARANTEED 2")
print("ORIGINAL VERSION")
print("*************************** AVERAGE OVER ALL FOLDS (misclassification costs = " + str(misclassificationCostsSymmetric) + ", sameClassCost = " + str(sameClassCost) + ") *******************************************")
evaluation.showHelper("total costs = ", testTotalCostsAllFolds)
evaluation.showHelper("feature costs = ", testFeatureCostsAllFolds)
evaluation.showHelper("misclassification costs = ", testMisClassificationCostsAllFolds)
evaluation.showHelper("accuracy = ", testAccuracyAllFolds)


print("densityRegressionModelName = ", densityRegressionModelName)
print("infoStr = ", infoStr)
print("Finished Successfully")


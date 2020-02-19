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

# dataName = "miniBooNE_5foldCV"
# dataName = "crab_5foldCV"
# dataName = "pima_5foldCV"
# dataName = "breastcancer_5foldCV"

# dataName = "pyhsioNetNoMissing_5foldCV"
# dataName = "pyhsioNetWithMissing_5foldCV"
dataName = "heartDiseaseWithMissing_5foldCV"

imputationMethod = "gaussian_imputation"


pooling = None
# pooling = "probabilityPooling"

# /opt/intel/intelpython3/bin/python evalClassificationAccuracyOnly.py

allTargetRecalls = [0.95, 0.99, 0.999]

NUMBER_OF_FOLDS = 5

# definedFeatureCosts = realdata.getFeaturesCosts(dataName)
# testAccuracyAllFoldsSVM = numpy.zeros(NUMBER_OF_FOLDS)

# testAccuracyAllFoldsLogistic = numpy.zeros(NUMBER_OF_FOLDS)
testAUCAllFoldsLogistic = numpy.zeros(NUMBER_OF_FOLDS)

# testAccuracyAllFoldsGP = numpy.zeros(NUMBER_OF_FOLDS)
testAUCAllFoldsGP = numpy.zeros(NUMBER_OF_FOLDS)

testRecallAllFoldsLogistic = []
testSpecifityAllFoldsLogistic = []
testFDRAllFoldsLogistic = []
testRecallAllFoldsLogistic_exactRecall = []
testFDRAllFoldsLogistic_exactRecall = []

testRecallAllFoldsGP = []
testSpecifityAllFoldsGP = []
testFDRAllFoldsGP = []
testRecallAllFoldsGP_exactRecall = []
testFDRAllFoldsGP_exactRecall = []
for i in range(len(allTargetRecalls)): 
    testRecallAllFoldsLogistic.append(numpy.zeros(NUMBER_OF_FOLDS))
    testSpecifityAllFoldsLogistic.append(numpy.zeros(NUMBER_OF_FOLDS))
    testFDRAllFoldsLogistic.append(numpy.zeros(NUMBER_OF_FOLDS))
    testRecallAllFoldsLogistic_exactRecall.append(numpy.zeros(NUMBER_OF_FOLDS))
    testFDRAllFoldsLogistic_exactRecall.append(numpy.zeros(NUMBER_OF_FOLDS))
    testRecallAllFoldsGP.append(numpy.zeros(NUMBER_OF_FOLDS))
    testSpecifityAllFoldsGP.append(numpy.zeros(NUMBER_OF_FOLDS))
    testFDRAllFoldsGP.append(numpy.zeros(NUMBER_OF_FOLDS))
    testRecallAllFoldsGP_exactRecall.append(numpy.zeros(NUMBER_OF_FOLDS))
    testFDRAllFoldsGP_exactRecall.append(numpy.zeros(NUMBER_OF_FOLDS))

for foldId in range(NUMBER_OF_FOLDS):
    
    print("process fold = ", foldId)
    
    
    if pooling != None:
        
        assert(False) # needs to be adjusted
    
        
        trainData, trainLabels, unlabeledData, testData, testLabels = realdata.loadSubsetBasic(dataName, None, foldId)
        trainLabels = experimentHelper.getLabelsStartingAtZero(trainLabels)
        testLabels = experimentHelper.getLabelsStartingAtZero(testLabels)
    
        # imputationMethod = "mice_imputation"
        # trainData = numpy.load(constants.BASE_FOLDER + dataName + "_" + imputationMethod + "_fold" + str(foldId) + "_trainData" + ".npy")
        
        stemFilenameForData = dataName + "_" + imputationMethod
        filename = constants.BASE_FOLDER + stemFilenameForData + "_fold" + str(foldId) + "_trainData"
        
        with open(filename, 'rb') as f:
            allImputedTrainData = pickle.load(f)
    
        assert(len(allImputedTrainData) == 5)
        
        print("run logistic regression")
        
        if pooling == "majorityVoting":
            allModels = []
            allThresholds = []
            for trainData in allImputedTrainData:
                bestLogRegModel, bestAlphaValue, _ = evaluation.getBestL2RegularizedLogisticRegressionModelNew(trainData, trainLabels)
                allThresholdLogistic = evaluation.getThresholdEstimate(trainData, trainLabels, bestAlphaValue, allTargetRecalls, modelType = "logReg")
                
                allModels.append(bestLogRegModel)
                allThresholds.append(allThresholdLogistic)
                
            print("pool all results")
            for targetRecallId in range(len(allTargetRecalls)):
                testRecallAllFoldsLogistic[targetRecallId][foldId], testFDRAllFoldsLogistic[targetRecallId][foldId] = evaluation.getPooledRecallAndFDR(allModels, testData, testLabels, allThresholds, targetRecallId)
            
        elif pooling == "probabilityPooling":
            allModels = []
            allBestHyperparameters = []
            for trainData in allImputedTrainData:
                bestLogRegModel, bestAlphaValue, _ = evaluation.getBestL2RegularizedLogisticRegressionModelNew(trainData, trainLabels)
                allModels.append(bestLogRegModel)
                allBestHyperparameters.append(bestAlphaValue)
                
            
            allThresholdLogistic = evaluation.getThresholdEstimate_pooled(allImputedTrainData, trainLabels, allBestHyperparameters, allTargetRecalls, modelType = "logReg")
            pooledPredictedProbs = evaluation.getPooledProbability(allModels, testData)
                        
            for i in range(len(allTargetRecalls)):
                testRecallAllFoldsLogistic[i][foldId] = evaluation.getRecall(testLabels, pooledPredictedProbs, allThresholdLogistic[i])
                testFDRAllFoldsLogistic[i][foldId] = evaluation.getFDR(testLabels, pooledPredictedProbs, allThresholdLogistic[i])
                
                exactThreshold = evaluation.getThresholdFromPredictedProbabilities(testLabels, pooledPredictedProbs, allTargetRecalls[i])
                testRecallAllFoldsLogistic_exactRecall[i][foldId] = evaluation.getRecall(testLabels, pooledPredictedProbs, exactThreshold)
                testFDRAllFoldsLogistic_exactRecall[i][foldId] = evaluation.getFDR(testLabels, pooledPredictedProbs, exactThreshold)
                
            testAUCAllFoldsLogistic[foldId] = sklearn.metrics.roc_auc_score(testLabels, pooledPredictedProbs)

        else:
            assert(False)

    
    
    else:
        
        trainData, trainLabels, unlabeledData, testData, testLabels = realdata.loadSubset(dataName, None, foldId, imputationMethod)

        print("run logistic regression")
        bestLogRegModel, bestAlphaValue, _ = evaluation.getBestL2RegularizedLogisticRegressionModelNew(trainData, trainLabels)
        allThresholdLogistic = evaluation.getThresholdEstimate(trainData, trainLabels, bestAlphaValue, allTargetRecalls, modelType = "logReg")
        
        for i in range(len(allTargetRecalls)):
            testRecallAllFoldsLogistic[i][foldId], testSpecifityAllFoldsLogistic[i][foldId], testFDRAllFoldsLogistic[i][foldId] = evaluation.getTestRecallSpecifityFDR(bestLogRegModel, testData, testLabels, allThresholdLogistic[i])
            testRecallAllFoldsLogistic_exactRecall[i][foldId], testFDRAllFoldsLogistic_exactRecall[i][foldId] = evaluation.getFDR_atExactRecall(bestLogRegModel, testData, testLabels, allTargetRecalls[i])
        
        testAUCAllFoldsLogistic[foldId] = evaluation.getTestAUC(bestLogRegModel, testData, testLabels)

        
#         print("run GP classifier")
#         bestGPClassifier, bestLengthScale = evaluation.getBestGP(trainData, trainLabels)
#         allThresholdGP = evaluation.getThresholdEstimate(trainData, trainLabels, bestLengthScale, allTargetRecalls, modelType = "GP")
#           
#         for i in range(len(allTargetRecalls)):
#             testRecallAllFoldsGP[i][foldId], testSpecifityAllFoldsGP[i][foldId], testFDRAllFoldsGP[i][foldId] = evaluation.getTestRecallSpecifityFDR(bestGPClassifier, testData, testLabels, allThresholdGP[i])
#             testRecallAllFoldsGP_exactRecall[i][foldId], testFDRAllFoldsGP_exactRecall[i][foldId] = evaluation.getFDR_atExactRecall(bestGPClassifier, testData, testLabels, allTargetRecalls[i])
#          
#         testAUCAllFoldsGP[foldId] = evaluation.getTestAUC(bestGPClassifier, testData, testLabels)



    
#     startTime = time.time()
#     covFuncRBF = sklearn.gaussian_process.kernels.RBF(length_scale = 1.0)
#     covFuncConst = sklearn.gaussian_process.kernels.ConstantKernel(constant_value=1.0)
#     covFuncFinal = sklearn.gaussian_process.kernels.Product(covFuncConst, covFuncRBF)
#     gpClassifier = sklearn.gaussian_process.GaussianProcessClassifier(kernel=covFuncFinal)
#     # gpClassifier = sklearn.gaussian_process.GaussianProcessClassifier(max_iter_predict=100)
#     gpClassifier.fit(trainData, trainLabels)
#     testRecallAllFoldsGP[foldId], testSpecifityAllFoldsGP[foldId] = evaluation.getTestRecallAndSpecifity(gpClassifier, testData, testLabels, thresholdLogistic)
#     testAccuracyAllFoldsGP[foldId] = evaluation.getTestAccuracy(gpClassifier, testData, testLabels)
#     testAUCAllFoldsGP[foldId] = evaluation.getTestAUC(gpClassifier, testData, testLabels)
#     print("runtime (in minutes) = " + str((time.time() - startTime) / 60.0))
#     
#     print("run SVM classifier")
#     startTime = time.time()
#     bestSVMRBFModel = evaluation.getBestSVMRBFModelAcc(trainData, trainLabels)
#     testAccuracyAllFoldsSVM[foldId] = evaluation.getTestAccuracy(bestSVMRBFModel, testData, testLabels)
#     print("runtime (in minutes) = " + str((time.time() - startTime) / 60.0))
    

print("")
print("AVERAGE RESULTS 5-FOLD CROSS-VAlIDATION")

print("*************************** AVERAGE ACCURACY OVER ALL FOLDS  *******************************************")
print("dataName = ", dataName)

print("*************************** Logistic regression  *******************************************")
for i in range(len(allTargetRecalls)):
    print("target recall = ", allTargetRecalls[i])
    evaluation.showHelperDetailed("    recall = ", testRecallAllFoldsLogistic[i])
    evaluation.showHelperDetailed("    FDR = ", testFDRAllFoldsLogistic[i])
    evaluation.showHelperDetailed("    specifity = ", testSpecifityAllFoldsLogistic[i])
    print("--- results at exact target recall: ---- ")
    evaluation.showHelperDetailed("    recall = ", testRecallAllFoldsLogistic_exactRecall[i])
    evaluation.showHelperDetailed("    FDR = ", testFDRAllFoldsLogistic_exactRecall[i])
    
evaluation.showHelperDetailed("auc = ", testAUCAllFoldsLogistic)

print("*************************** GP  *******************************************")
for i in range(len(allTargetRecalls)):
    print("target recall = ", allTargetRecalls[i])
    evaluation.showHelperDetailed("    recall = ", testRecallAllFoldsGP[i])
    evaluation.showHelperDetailed("    FDR = ", testFDRAllFoldsGP[i])
    evaluation.showHelperDetailed("    specifity = ", testSpecifityAllFoldsGP[i])
    print("--- results at exact target recall: ---- ")
    evaluation.showHelperDetailed("    recall = ", testRecallAllFoldsGP_exactRecall[i])
    evaluation.showHelperDetailed("    FDR = ", testFDRAllFoldsGP_exactRecall[i])
evaluation.showHelperDetailed("auc = ", testAUCAllFoldsGP)


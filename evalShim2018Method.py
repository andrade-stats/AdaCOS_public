import numpy
import evaluation
import sklearn.metrics

def getAllStatistics(filename):
    allInfoArray = numpy.load(filename)
    numberOfCovariates = allInfoArray.shape[0] - 2
    covariateUsage = allInfoArray[:, 2:allInfoArray.shape[0]]
    
    labels = allInfoArray[:,0]
    predictedProbTrueLabel = allInfoArray[:,1]
    return labels, predictedProbTrueLabel, covariateUsage


# dataName = "pima_5foldCV"
# dataName = "breastcancer_5foldCV"
dataName = "pyhsioNetWithMissing_5foldCV"
# dataName = "heartDiseaseWithMissing_5foldCV"

allRCOSTS = numpy.linspace(start=0.0, stop=-0.1, num = 11)

# allTargetRecalls = [0.95, 0.99, 0.999]
allTargetRecalls = [0.95]


BASEFOLDER = "/export/home/s-andrade/eclipseWorkspace/Joint-AFA-Classification/results/"
NUMBER_OF_FOLDS = 5


for featureCosts in allRCOSTS:
    
    testAUCAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    testNrFeaturesAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    
    testRecallAllFolds = []
    testSpecifityAllFolds = []
    testFDRAllFolds = []
    
    
    for i in range(len(allTargetRecalls)): 
        testRecallAllFolds.append(numpy.zeros(NUMBER_OF_FOLDS))
        testSpecifityAllFolds.append(numpy.zeros(NUMBER_OF_FOLDS))
        testFDRAllFolds.append(numpy.zeros(NUMBER_OF_FOLDS))


    for foldId in range(NUMBER_OF_FOLDS):
        FILENAME_STEM = BASEFOLDER + dataName + "_fold" + str(foldId) + "/" + "costs" + str(featureCosts) + "_"
        labelsValidation, allPredictedProbsValidation, covariateUsageValidation = getAllStatistics(FILENAME_STEM + "val.npy")
        labelsTest, allPredictedProbsTest, covariateUsageTest = getAllStatistics(FILENAME_STEM + "ts.npy")
       
        for i, targetRecall in enumerate(allTargetRecalls):
            threshold = evaluation.getThresholdFromPredictedProbabilities(labelsValidation, allPredictedProbsValidation, targetRecall)
            testRecallAllFolds[i][foldId] = evaluation.getRecall(labelsTest, allPredictedProbsTest, threshold)
            testSpecifityAllFolds[i][foldId] = evaluation.getSpecifity(labelsTest, allPredictedProbsTest, threshold)
            testFDRAllFolds[i][foldId] = evaluation.getFDR(labelsTest, allPredictedProbsTest, threshold)
            
        testAUCAllFolds[foldId] = sklearn.metrics.roc_auc_score(labelsTest, allPredictedProbsTest)
        testNrFeaturesAllFolds[foldId] = numpy.mean(numpy.sum(covariateUsageTest, axis = 1))
          
        # print("featureCosts = ", featureCosts)
        # print("testAUCAllFolds[foldId] = ", testAUCAllFolds[foldId])
        # print("average number of features used = ", avgNumberOfUsedCovariates)
    


    print("")
    print("AVERAGE RESULTS 5-FOLD CROSS-VAlIDATION")
    print("featureCosts = ", featureCosts)
    print("*************************** AVERAGE ACCURACY OVER ALL FOLDS  *******************************************")
    print("dataName = ", dataName)
    
    print("*************************** Shim 2018 et al method  *******************************************")
    for i in range(len(allTargetRecalls)):
        print("target recall = ", allTargetRecalls[i])
        evaluation.showHelperDetailed("    recall = ", testRecallAllFolds[i])
        evaluation.showHelperDetailed("    FDR = ", testFDRAllFolds[i])
        evaluation.showHelperDetailed("    specifity = ", testSpecifityAllFolds[i])
    evaluation.showHelperDetailed("auc = ", testAUCAllFolds)
    evaluation.showHelperDetailed("nr features = ", testNrFeaturesAllFolds)






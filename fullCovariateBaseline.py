import numpy
import realdata
import sklearn.linear_model
import sklearn.metrics
import evaluation
import experimentSetting
import experimentHelper
import prepareFeatureSets
import sklearn.gaussian_process



# dataName = "miniBooNE_5foldCV"
dataName = "pima_5foldCV"
# dataName = "breastcancer_5foldCV"
# dataName = "pyhsioNetNoMissing_5foldCV"
# dataName = "heartDiseaseWithMissing_5foldCV"

# sudo -E /opt/intel/intelpython3/bin/pip install group-lasso
# /opt/intel/intelpython3/bin/python fullCovariateResults.py

imputationMethod = "gaussian_imputation"

allMisclassificationCostsSymmetric = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]

resultsRecorder = experimentHelper.ResultsRecorder(len(allMisclassificationCostsSymmetric))

for misclassificationCostsSymmetric in allMisclassificationCostsSymmetric:

    numpy.random.seed(3523421)
        
    if dataName == "pima_5foldCV":
        sameClassCost = -50.0 # set to -50.0 in order to compare to Li and Carin work
        # assert(misclassificationCostsSymmetric == 400 or misclassificationCostsSymmetric == 800)
    else:
        sameClassCost = 0.0
    
    
    # row id = true class id, column id = predicted class id
    misclassificationCosts = numpy.zeros((2,2))
    misclassificationCosts[0, 1] = misclassificationCostsSymmetric 
    misclassificationCosts[1, 0] = misclassificationCostsSymmetric
    misclassificationCosts[0, 0] = sameClassCost
    misclassificationCosts[1, 1] = sameClassCost
    
    NUMBER_OF_FOLDS = 5
    
    definedFeatureCosts = realdata.getFeaturesCosts(dataName)
    
    testTotalCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    testFeatureCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    testMisClassificationCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    testAccuracyAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    testAUCAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    
    for foldId in range(NUMBER_OF_FOLDS):
        
        trainData, trainLabels, unlabeledData, testData, testLabels = realdata.loadSubset(dataName, None, foldId, imputationMethod)
        
        selectedFeatureIds = numpy.arange(trainData.shape[1])
        
        # allFeatureArraysInOrder, _ = prepareFeatureSets.getAllFeatureSetsInOrderWithL1LogReg(trainData, trainLabels, unlabeledData, None, definedFeatureCosts)
        allFeatureArraysInOrder = evaluation.nonLinearFeatureSelection_withGAM(trainData, trainLabels, definedFeatureCosts)
        allFeatureArraysInOrder = prepareFeatureSets.filterToEnsureSetInclusionOrder(allFeatureArraysInOrder)
        print("found covariate sets = ")
        for i in range(len(allFeatureArraysInOrder)):
            print("covariateIds = " + str(allFeatureArraysInOrder[i]) + " | expected total costs =  ?")
    
        assert(False)
        
        # bestLogRegModel, _, bestAlphaValue = evaluation.getBestL2RegularizedLogisticRegressionModelNew(trainData, trainLabels)
        
        print("****** LOGISTIC REGRESSION RESULTS ********")
        testTotalCostsAllFolds[foldId], testFeatureCostsAllFolds[foldId], testMisClassificationCostsAllFolds[foldId], testAccuracyAllFolds[foldId] = evaluation.getTestDataAverageTotalCostsWithFixedFeatureIds(bestModel, testData, testLabels, misclassificationCosts, selectedFeatureIds, definedFeatureCosts)
        testAUCAllFolds[foldId] = evaluation.getTestAUC(bestModel, testData[:, selectedFeatureIds], testLabels)
    
        # gpClassifier = sklearn.gaussian_process.GaussianProcessClassifier(max_iter_predict=100)
        # gpClassifier.fit(trainData, trainLabels)
        # print("****** GP RESULTS ********")
        # allTestMissclassificationCostsGP[foldId], allTestAccuraciesGP[foldId] = evaluation.getTestDataPerformance(gpClassifier, testData, testLabels, misclassificationCosts)
    
        # bestSVMRBFModel = evaluation.getBestSVMRBFModel(trainData, trainLabels, misclassificationCosts)
        # print("****** SVM-RBF RESULTS ********")
        # allTestMissclassificationCostsSVMRBF[foldId], allTestAccuraciesSVMRBF[foldId] = evaluation.getTestDataPerformance(bestSVMRBFModel, testData, testLabels, misclassificationCosts)
         
    
    print("")
    print("AVERAGE RESULTS 5-FOLD CROSS-VAlIDATION")
    
    print("*************************** AVERAGE OVER ALL FOLDS (misclassification costs = " + str(misclassificationCostsSymmetric) + ", sameClassCost = " + str(sameClassCost) + ") *******************************************")
    evaluation.showHelper("total costs = ", testTotalCostsAllFolds)
    evaluation.showHelper("feature costs = ", testFeatureCostsAllFolds)
    evaluation.showHelper("misclassification costs = ", testMisClassificationCostsAllFolds)
    evaluation.showHelper("accuracy = ", testAccuracyAllFolds)
    evaluation.showHelper("AUC = ", testAUCAllFolds)
    
    resultsRecorder.addAll(misclassificationCostsSymmetric, (testTotalCostsAllFolds, testFeatureCostsAllFolds, testMisClassificationCostsAllFolds, testAccuracyAllFolds, testAUCAllFolds))


resultsRecorder.writeOutResults(dataName + "_fullCovariateResults")

import numpy
import realdata
import sklearn.metrics
import evaluation
import experimentHelper
import dynamicAcquisition
import time
import sys
import prepareFeatureSets
import preprocessing
import constants
import pickle


# /opt/intel/intelpython3/bin/python precalculateAllGreedyFeatureSets.py pima_5foldCV Combined symmetricCost
# /opt/intel/intelpython3/bin/python precalculateAllGreedyFeatureSets.py pima_5foldCV Combined asymmetricCost
# /opt/intel/intelpython3/bin/python precalculateAllGreedyFeatureSets.py heartDiseaseWithMissing_5foldCV Combined asymmetricCost NonLinearl2
# /opt/intel/intelpython3/bin/python precalculateAllGreedyFeatureSets.py breastcancer_5foldCV Combined asymmetricCost

dataName = sys.argv[1]

classificationModelName = sys.argv[2]
assert(classificationModelName == "logReg" or classificationModelName == "GAM" or classificationModelName == "Combined")

# only used to compare with the results from previous work on Diabetes data
assert(sys.argv[3] == "asymmetricCost" or sys.argv[3] == "symmetricCost")
USE_SYMMETRIC_MISCLASSIFICATION_COSTS = sys.argv[3] == "symmetricCost"






if USE_SYMMETRIC_MISCLASSIFICATION_COSTS:
    assert(dataName == "pima_5foldCV")
    allTargetRecalls = [None]
    ALL_FALSE_POSITIVE_COSTS = [400,800]
    costTypeString = "symmetricCost"
else:
    ALL_FALSE_POSITIVE_COSTS = constants.allFalsePositiveCosts
    FN_TO_FP_RATIO = 10.0
    costTypeString = "asymmetricCost"
    

definedFeatureCosts = realdata.getFeaturesCosts(dataName)


for falsePositiveCost in ALL_FALSE_POSITIVE_COSTS:
    
    if USE_SYMMETRIC_MISCLASSIFICATION_COSTS:
        falseNegativeCost = falsePositiveCost
        assert(dataName == "pima_5foldCV")
        correctClassificationCost = -50.0 # in order to align to the setting in (Ji and Carin, 2007; Dulac-Arnold et al., 2012) for Diabetes.
        
        misclassificationCosts = numpy.zeros((2, 2))
        misclassificationCosts[0, 1] = falsePositiveCost 
        misclassificationCosts[1, 0] = falseNegativeCost 
        misclassificationCosts[0, 0] = correctClassificationCost
        misclassificationCosts[1, 1] = correctClassificationCost
    else:
        falseNegativeCost = falsePositiveCost * FN_TO_FP_RATIO
        misclassificationCosts = numpy.zeros((2, 2))
        misclassificationCosts[0, 1] = falsePositiveCost 
        misclassificationCosts[1, 0] = falseNegativeCost 
        misclassificationCosts[0, 0] = 0.0
        misclassificationCosts[1, 1] = 0.0
        
                    
    MODEL_FOLDERNAME =  "/export/home/s-andrade/newStart/eclipseWorkspaceDynamic/DynamicCovariateSelection/models/"
    trainedModelsFilename = dataName + "_" + classificationModelName + "_" + costTypeString + "_" + str(falsePositiveCost) + "_greedy"
    
    allPredictionModels_allFolds = []
    allTrainingTrueProbsAllModels_allFolds = []
    allFeatureArraysInOrder_allFolds = []


    for foldId in range(constants.NUMBER_OF_FOLDS):
        trainData, trainLabels, unlabeledData, testData, testLabels = realdata.loadSubset(dataName, None, foldId, constants.IMPUTATION_METHOD)
        allTrainingTrueProbsAllModels, allPredictionModels, allFeatureArraysInOrder, _ = prepareFeatureSets.getAllFeatureSetsInOrderWithGreedyMethod_normal(trainData, trainLabels, misclassificationCosts, definedFeatureCosts, classificationModelName, falsePositiveCost = None, targetRecall = None)

        allPredictionModels_allFolds.append(allPredictionModels)
        allTrainingTrueProbsAllModels_allFolds.append(allTrainingTrueProbsAllModels)
        allFeatureArraysInOrder_allFolds.append(allFeatureArraysInOrder)
    
    with open(MODEL_FOLDERNAME + trainedModelsFilename + "_models", "wb") as f:
        pickle.dump(allPredictionModels_allFolds,f)
    with open(MODEL_FOLDERNAME + trainedModelsFilename + "_probs", "wb") as f:
        pickle.dump(allTrainingTrueProbsAllModels_allFolds,f)
    with open(MODEL_FOLDERNAME + trainedModelsFilename + "_features", "wb") as f:
        pickle.dump(allFeatureArraysInOrder_allFolds,f)       


print("saved all successfully: " + dataName + "_" + classificationModelName + "_" + costTypeString)


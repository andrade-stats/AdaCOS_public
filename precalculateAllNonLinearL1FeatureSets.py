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


# /data/intelpython3/bin/python precalculateAllNonLinearL1FeatureSets.py pima_5foldCV Combined 
# /opt/intel/intelpython3/bin/python precalculateAllNonLinearL1FeatureSets.py breastcancer_5foldCV Combined
# /opt/intel/intelpython3/bin/python precalculateAllNonLinearL1FeatureSets.py heartDiseaseWithMissing_5foldCV Combined 

# currently running on Manto54:
# /opt/intel/intelpython3/bin/python precalculateAllNonLinearL1FeatureSets.py pyhsioNetWithMissing_5foldCV Combined


dataName = sys.argv[1]

classificationModelName = sys.argv[2]
assert(classificationModelName == "logReg" or classificationModelName == "GAM" or classificationModelName == "Combined")




    

definedFeatureCosts = realdata.getFeaturesCosts(dataName)

                    
MODEL_FOLDERNAME =  "/export/home/s-andrade/newStart/eclipseWorkspaceDynamic/DynamicCovariateSelection/models/"
trainedModelsFilename = dataName + "_" + classificationModelName + "_nonLinearL1"

allPredictionModels_allFolds = []
allTrainingTrueProbsAllModels_allFolds = []
allFeatureArraysInOrder_allFolds = []


for foldId in range(constants.NUMBER_OF_FOLDS):
    trainData, trainLabels, unlabeledData, testData, testLabels = realdata.loadSubset(dataName, None, foldId, constants.IMPUTATION_METHOD)

    allFeatureArraysInOrder = evaluation.nonLinearFeatureSelection_withGAM(trainData, trainLabels, definedFeatureCosts)
    allFeatureArraysInOrder = prepareFeatureSets.filterToEnsureSetInclusionOrder(allFeatureArraysInOrder)
    allPredictionModels, allTrainingTrueProbsAllModels = prepareFeatureSets.getAllBestModelsAndTrainingTrueProbs(trainData, trainLabels, allFeatureArraysInOrder, classificationModelName)

    allPredictionModels_allFolds.append(allPredictionModels)
    allTrainingTrueProbsAllModels_allFolds.append(allTrainingTrueProbsAllModels)
    allFeatureArraysInOrder_allFolds.append(allFeatureArraysInOrder)

with open(MODEL_FOLDERNAME + trainedModelsFilename + "_models", "wb") as f:
    pickle.dump(allPredictionModels_allFolds,f)
with open(MODEL_FOLDERNAME + trainedModelsFilename + "_probs", "wb") as f:
    pickle.dump(allTrainingTrueProbsAllModels_allFolds,f)
with open(MODEL_FOLDERNAME + trainedModelsFilename + "_features", "wb") as f:
    pickle.dump(allFeatureArraysInOrder_allFolds,f)       


print("saved all nonLinearL1 successfully: " + dataName + "_" + classificationModelName)


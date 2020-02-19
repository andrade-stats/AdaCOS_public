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

# /opt/intel/intelpython3/bin/python getOptimalSequence_recall.py pima_5foldCV Combined
# /opt/intel/intelpython3/bin/python getOptimalSequence_recall.py breastcancer_5foldCV Combined
# /opt/intel/intelpython3/bin/python getOptimalSequence_recall.py heartDiseaseWithMissing_5foldCV Combined
# /opt/intel/intelpython3/bin/python getOptimalSequence_recall.py pyhsioNetWithMissing_5foldCV Combined

# dataName = "pima_5foldCV" 
# dataName = "breastcancer_5foldCV"
dataName = "heartDiseaseWithMissing_5foldCV"
# dataName = "pyhsioNetWithMissing_5foldCV"

USE_UNLABELED_DATA = False
# assert(sys.argv[2] == "USE_UNLABELED_DATA" or sys.argv[2] == "NO_UNLABELED_DATA")
# USE_UNLABELED_DATA = (sys.argv[2] == "USE_UNLABELED_DATA")


onlyLookingOneStepAhead = False
NR_OF_USED_CPUS = None
NUMBER_OF_SAMPLES = None
    
classificationModelName = "Combined"

 



FEATURE_SELECTION_METHOD = "nonLinearL1" 






definedFeatureCosts = realdata.getFeaturesCosts(dataName)

print("number of features = ", definedFeatureCosts.shape[0])

MODEL_FOLDERNAME =  "/export/home/s-andrade/newStart/eclipseWorkspaceDynamic/DynamicCovariateSelection/models/"

trainedModelsFilenameNonLinearL1 = dataName + "_" + classificationModelName + "_nonLinearL1"

with open(MODEL_FOLDERNAME + trainedModelsFilenameNonLinearL1 + "_features", "rb") as f:
    allFeatureArraysInOrderNonLinearL1_allFolds = pickle.load(f)


# for foldId in range(constants.NUMBER_OF_FOLDS):
#     print("foldId = ", foldId)
#     allFeatureArraysInOrder = allFeatureArraysInOrderNonLinearL1_allFolds[foldId]
#     
#     for i in range(len(allFeatureArraysInOrder)):
#         print(allFeatureArraysInOrder[i])
# 
#     assert(len(allFeatureArraysInOrder) == definedFeatureCosts.shape[0] + 1)
#     print("number of different feature sets = ", len(allFeatureArraysInOrder))
# 
# print("PASSED ALL")



for foldId in range(constants.NUMBER_OF_FOLDS):
    trainData, trainLabels, unlabeledData, testData, testLabels = realdata.loadSubset(dataName, None, foldId, constants.IMPUTATION_METHOD)
    print("number of covariates = ", trainData.shape[1])
    print("total number of samples = ", (trainData.shape[0] + testData.shape[0]))


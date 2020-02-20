import scipy.io
import numpy
import realdata
import sklearn.metrics
import sklearn.model_selection
import evaluation
import constants
import experimentSettingBaselines
import sys

# /export/home/s-andrade/newStart/eclipseWorkspaceDynamic/DynamicCovariateSelection
# /opt/intel/intelpython3/bin/python prepareForMatlabBaselines.py pyhsioNetWithMissing_5foldCV
# /opt/intel/intelpython3/bin/python prepareForMatlabBaselines.py pima_5foldCV
# /opt/intel/intelpython3/bin/python prepareForMatlabBaselines.py heartDiseaseWithMissing_5foldCV
# /opt/intel/intelpython3/bin/python prepareForMatlabBaselines.py breastcancer_5foldCV

# dataName = "pima_5foldCV"
# dataName = "breastcancer_5foldCV"
# dataName = "heartDiseaseWithMissing_5foldCV"
# dataName =  "pyhsioNetWithMissing_5foldCV"

dataName = sys.argv[1]
# dataName = "pima_5foldCV"


classificationModelName = "Combined"
# classificationModelName = "logReg"




featureCosts = realdata.getFeaturesCosts(dataName)


for foldId in range(constants.NUMBER_OF_FOLDS):

    print("prepare foldId = ", foldId)
    
    # xtr: training data, dimension = # training examples x # features
    # xtv: validation data, dimension = # validation examples x # features
    # xte: test data, dimension = # test examples x # features
    # ytr: class label. -1/1 for binary classification, dimension = # training examples x 1
    # ytv: class label. -1/1 for binary classification, dimension = # validation examples x 1
    # yte: class label. -1/1 for binary classification, dimension = # test examples x 1
    # cost: feature acquisition cost vector, dimension = # features x 1
       
    trainAndEvalData, trainAndEvalLabels, unlabeledData, testData, testLabels = realdata.loadSubset(dataName, None, foldId, constants.IMPUTATION_METHOD)
     
    NR_SPLITS = experimentSettingBaselines.NUMBER_OF_TRAIN_EVAL_FOLDS_MATLAB_METHODS * 2
    assert(NR_SPLITS == 10)
    kFoldMaster = sklearn.model_selection.StratifiedKFold(n_splits = NR_SPLITS,  shuffle=True, random_state=4324232)
    for trainEvalSplitNr, (train_index, eval_index) in enumerate(kFoldMaster.split(trainAndEvalData, trainAndEvalLabels)):
        trainData = trainAndEvalData[train_index]
        trainLabels = trainAndEvalLabels[train_index]
        evalData = trainAndEvalData[eval_index]
        evalLabels = trainAndEvalLabels[eval_index]

        outputFilename = experimentSettingBaselines.MATLAB_FOLDER_DATA + dataName + "_" + str(trainEvalSplitNr) + "trainEvalSplitNr_" + str(foldId)
        evaluation.prepareForMatlab(trainData, trainLabels, evalData, evalLabels, featureCosts, outputFilename, classificationModelName)
    
    
    outputFilename = experimentSettingBaselines.MATLAB_FOLDER_DATA + dataName + "_forFinalTrainingAndTesting_" + str(foldId)
    evaluation.prepareForMatlab(trainAndEvalData, trainAndEvalLabels, testData, testLabels, featureCosts, outputFilename, classificationModelName)
    

print("FINISHED:")
print(dataName)
print(classificationModelName)

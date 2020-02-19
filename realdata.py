import re
import numpy
import sklearn.model_selection
import preprocessing
import sklearn.datasets

import constants
import physioNet
import experimentHelper

BASE_FOLDER = "/export/home/s-andrade/newStart/dynamicCovariateBaselines/datasets/"

# loads the crab data as used in "Gaussian processes for Bayesian classification via hybrid Monte Carlo"
def loadCrabData():
    filename = BASE_FOLDER + "crabs.csv"
    
    numberOfVariables = 5
    
    labels = numpy.zeros(200)
    data = numpy.zeros((200, numberOfVariables))
    
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            values = line.split(",")
            if i == 0:
                variableNames = values[3:len(values)]
            else:
                assert(values[1] == "M" or values[1] == "F")
                if values[1] == "M":
                    labels[i - 1] = -1
                else:
                    labels[i - 1] = 1
                
                data[i - 1] = [float(val) for val in values[3:len(values)]]
                
    
    
    # print("variableNames = ")
    # print(variableNames) 
    # print("data = ")
    # print(data)
    # print("labels = ")
    # print(labels)
    
    costs = numpy.ones(data.shape[1])
        
    assert(labels.shape[0] == data.shape[0])
    assert(data.shape[1] == costs.shape[0])
    
    return data, labels, costs, variableNames


def prepareCrabData():
    numpy.random.seed(3523421)   
    data, labels, costs, variableNames = loadCrabData()
    
    stemFilename = "crab_5foldCV" 
    save5FoldCV_noUnlabeledData(data, labels, stemFilename)
    return   



def loadPimaDiabetesData():
    filename = BASE_FOLDER +  "diabetes.csv"

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            columnNames = line.split(",")
            break
    
    
    
    dataWithLabels = numpy.loadtxt(filename, delimiter=",", skiprows=1)
    
    numberOfVariables = dataWithLabels.shape[1] - 1
    
    labels = dataWithLabels[:, numberOfVariables]
    data = dataWithLabels[:,0:numberOfVariables]
    
    for label in labels:
        assert(label == 1.0 or label == 0.0)
    
    # in dollars adapted from "Cost-sensitive Feature Acquisition and Classification"
    costs = numpy.asarray([1.0, 17.61, 1.0, 1.0, 22.78, 1.0, 1.0, 1.0])
    
    assert(costs.shape[0] == numberOfVariables)
    assert(labels.shape[0] == data.shape[0])
    assert(data.shape[1] == costs.shape[0])
    
    # replace 0 -> -1 (to ensure that labels are -1/1 as required by ADAPT GRAD
    labels[labels == 0.0] = -1.0 
    
    variableNames = columnNames[0:numberOfVariables]
    
    return data, labels, costs, variableNames
    

def loadMiniBooNEData():
    filename = BASE_FOLDER + "MiniBooNE_PID.txt"
    
    labels = None
    data = None
    
    with open(filename, "r") as f:
        number_true_samples = None
        number_false_samples = None
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0:
                number_true_samples = int(line.split(" ")[0])
                number_false_samples = int(line.split(" ")[1]) 
                # print("numer_true_samples = ", number_true_samples)
                # print("numer_false_samples = ", number_false_samples)
                labels = numpy.zeros(number_true_samples + number_false_samples)
                data = numpy.zeros((number_true_samples + number_false_samples, 50))
            else:
                vecAsStr = re.split(" +", line)
                assert(len(vecAsStr) == 50)
                vec = [float(s) for s in vecAsStr]
                data[i-1] = vec
                if i <= number_true_samples:
                    labels[i-1] = 1
                else:
                    labels[i-1] = -1
    
    costs = numpy.ones(data.shape[1])
        
    assert(labels.shape[0] == data.shape[0])
    assert(data.shape[1] == costs.shape[0])
    
    return data, labels, costs


# from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
# explanation in https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)
def loadWisconsinBreastCancer():
    filename = BASE_FOLDER + "breast-cancer"

    dataWithLabels = sklearn.datasets.load_svmlight_file(filename)
    
    data = dataWithLabels[0].todense()
    labels = dataWithLabels[1] - 3 
    labels = labels.astype(numpy.int)
    
    costs = numpy.ones(data.shape[1])
        
    assert(labels.shape[0] == data.shape[0])
    assert(data.shape[1] == costs.shape[0])
    
    variableNames = ["Sample Code", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size",
                     "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"]
    
    assert(len(variableNames) == costs.shape[0])
    return data, labels, costs, variableNames


def getFeaturesCosts(dataName):
    if dataName.startswith("miniBooNE"):
        data, labels, costs = loadMiniBooNEData()
    elif dataName.startswith("pima_"):
        data, labels, costs, variableNames = loadPimaDiabetesData()
    elif dataName.startswith("pimaUniform_"):
        costs = numpy.ones(8)
    elif dataName.startswith("breastcancer"):
        data, labels, costs, variableNames = loadWisconsinBreastCancer()
    elif dataName.startswith("crab"):
        data, labels, costs, variableNames = loadCrabData()
    elif dataName.startswith("pyhsioNetNoMissing_"):
        costs = numpy.ones(16)
    elif dataName.startswith("pyhsioNetWithMissing_"):
        costs = numpy.ones(30)
    elif dataName.startswith("heartDiseaseWithMissing_"):
        costs = numpy.asarray([1.0, 1.0, 1.0, 1.0, 7.27, 5.20, 15.50, 102.90, 87.30, 87.30, 87.30, 100.90, 102.90])
    else:
        assert(False)
        
    return costs


def getVariableNames(dataName):
    if dataName == "miniBooNE":
        assert(False)
    elif dataName == "pima":
        data, labels, costs, variableNames = loadPimaDiabetesData()
    elif dataName == "breastcancer":
        data, labels, costs, variableNames = loadWisconsinBreastCancer()
    elif dataName.startswith("crab"):
        data, labels, costs, variableNames = loadCrabData()
    elif dataName.startswith("pyhsioNetNoMissing_"):   
        sortedIds = numpy.sort(list(physioNet.VARIABLE_SUBSET_FOR_EVALUATION_NO_MISSING))
        return numpy.asarray(physioNet.allAttributeNames)[sortedIds]
    elif dataName.startswith("pyhsioNetWithMissing_"):   
        sortedIds = numpy.sort(list(physioNet.VARIABLE_SUBSET_FOR_EVALUATION_WITH_MISSING))
        return numpy.asarray(physioNet.allAttributeNames)[sortedIds]
    else:
        assert(False)
        
    return numpy.asarray(variableNames)


# checked
def getRandomSplitsNew(data, labels, trainSize, testSize, unlabeledSize, NUMBER_OF_RANDOM_SPLITS = 5):
    assert(trainSize >= 10)
    assert(testSize >= 10)
    assert(unlabeledSize == "remaining" or unlabeledSize >= 10)
    
    if unlabeledSize == "remaining":
        unlabeledSize = labels.shape[0] - (trainSize + testSize)
    
    allTrainData = []
    allTrainLabels = []
    
    allUnlabeledData = []
    
    allTestData = []
    allTestLabels = []
    
    shufflerTrainTestAndRest = sklearn.model_selection.StratifiedShuffleSplit(n_splits=NUMBER_OF_RANDOM_SPLITS, train_size=trainSize+testSize, test_size=unlabeledSize, random_state=382922)
    shufflerTrainTest = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, train_size=trainSize, test_size=testSize, random_state=382922)
    for trainTest_index, unlabeled_index in shufflerTrainTestAndRest.split(data, labels):
        allUnlabeledData.append(data[unlabeled_index])
        
        trainTestData = data[trainTest_index]
        trainTestLabels = labels[trainTest_index]
        assert(trainTestData.shape[0] == trainSize + testSize)
        for train_index, test_index in shufflerTrainTest.split(trainTestData, trainTestLabels):
            allTrainData.append(trainTestData[train_index])
            allTrainLabels.append(trainTestLabels[train_index])
            allTestData.append(trainTestData[test_index])
            allTestLabels.append(trainTestLabels[test_index])
    
    assert(len(allTrainData) == NUMBER_OF_RANDOM_SPLITS and len(allTestData) == NUMBER_OF_RANDOM_SPLITS and len(allUnlabeledData) == NUMBER_OF_RANDOM_SPLITS)  
    return allTrainData, allTrainLabels, allUnlabeledData, allTestData, allTestLabels



def save5FoldCV_noUnlabeledData(data, labels, filenameStem):
    
    filenameStemWithPath = constants.BASE_FOLDER + filenameStem 
    
    NUMBER_OF_FOLDS = 5
    
    foldId = 0
    cv = sklearn.model_selection.StratifiedKFold(n_splits=NUMBER_OF_FOLDS, shuffle=True, random_state=564343)
    for train_index, test_index in cv.split(data, labels):
        numpy.save(filenameStemWithPath + "_fold" + str(foldId) + "_trainData", data[train_index])
        numpy.save(filenameStemWithPath + "_fold" + str(foldId) + "_trainLabels", numpy.asarray(labels[train_index], dtype = numpy.int))
        numpy.save(filenameStemWithPath + "_fold" + str(foldId) + "_testData", data[test_index])
        numpy.save(filenameStemWithPath + "_fold" + str(foldId) + "_testLabels", numpy.asarray(labels[test_index], dtype = numpy.int))
        
        unlabeledData = numpy.zeros((0, data.shape[1]))
        numpy.save(filenameStemWithPath + "_fold" + str(foldId) + "_unlabeledData", unlabeledData)
        
        foldId += 1
        
    return



def showStats(labels):
    print("true/false labels = " + str(numpy.count_nonzero(labels == 1)) + ", " + str(numpy.count_nonzero(labels == -1)))
    return


def saveAllSubsetsNew(allTrainData, allTrainLabels, allUnlabeledData, allTestData, allTestLabels, filenameWithTr):
    
    for foldId in range(len(allTrainData)):
        numpy.save(constants.BASE_FOLDER + filenameWithTr + "_fold" + str(foldId) + "_trainData", allTrainData[foldId])
        numpy.save(constants.BASE_FOLDER + filenameWithTr + "_fold" + str(foldId) + "_trainLabels", numpy.asarray(allTrainLabels[foldId], dtype = numpy.int))
    
        numpy.save(constants.BASE_FOLDER + filenameWithTr + "_fold" + str(foldId) + "_unlabeledData", allUnlabeledData[foldId])
        
        numpy.save(constants.BASE_FOLDER + filenameWithTr + "_fold" + str(foldId) + "_testData", allTestData[foldId])
        numpy.save(constants.BASE_FOLDER + filenameWithTr + "_fold" + str(foldId) + "_testLabels", numpy.asarray(allTestLabels[foldId], dtype = numpy.int))
    
    return


 
def addInteractionTerms(data):
    p = data.shape[1]
    lowerTriangleIds = numpy.tril_indices(p) # including diagonal
    
    numberOfInteractionTerms = int(((p * (p-1)) / 2) + p) 
    newP = p + numberOfInteractionTerms
    newData = numpy.zeros((data.shape[0], newP))
    newData[:, 0:p] = data
    
    for i in range(data.shape[0]):
        x = data[i]
        repX = numpy.tile(x, (p,1)).transpose()
        allInteractionsMatrix = repX @ numpy.diag(x)
        allInteractionsVec = allInteractionsMatrix[lowerTriangleIds]
        assert(allInteractionsVec.shape[0] == numberOfInteractionTerms)
        newData[i, p:newP] = allInteractionsVec
    
    return newData
    
# data = numpy.zeros((2,3))
# data[0] = [1,2,3]
# data[1] = [0.1,0.2,0.3]
# newData = addInteractionTerms(data)
# print("newData = ")
# print(newData)    

# also standardizes the data
def loadSubsetBasic(stemFilename, trainSize, foldId):
    
    if stemFilename.startswith("pimaUniform_5foldCV"):
        stemFilenameForData = "pima_5foldCV"
        trainData = numpy.load(constants.BASE_FOLDER + stemFilenameForData + "_fold" + str(foldId) + "_trainData" + ".npy")
        trainLabels = numpy.load(constants.BASE_FOLDER + stemFilenameForData + "_fold" + str(foldId) + "_trainLabels" + ".npy")
        testData = numpy.load(constants.BASE_FOLDER + stemFilenameForData + "_fold" + str(foldId) + "_testData" + ".npy")
        testLabels = numpy.load(constants.BASE_FOLDER + stemFilenameForData + "_fold" + str(foldId) + "_testLabels" + ".npy")
        unlabeledData = numpy.load(constants.BASE_FOLDER + stemFilenameForData + "_fold" + str(foldId) + "_unlabeledData" + ".npy")
    
    elif stemFilename.endswith("5foldCV"):
        trainData = numpy.load(constants.BASE_FOLDER + stemFilename + "_fold" + str(foldId) + "_trainData" + ".npy")
        trainLabels = numpy.load(constants.BASE_FOLDER + stemFilename + "_fold" + str(foldId) + "_trainLabels" + ".npy")
        testData = numpy.load(constants.BASE_FOLDER + stemFilename + "_fold" + str(foldId) + "_testData" + ".npy")
        testLabels = numpy.load(constants.BASE_FOLDER + stemFilename + "_fold" + str(foldId) + "_testLabels" + ".npy")
        unlabeledData = numpy.load(constants.BASE_FOLDER + stemFilename + "_fold" + str(foldId) + "_unlabeledData" + ".npy")
        
    else:
        filenameWithTr = stemFilename + "_" + str(trainSize) + "tr" 
        trainData = numpy.load(constants.BASE_FOLDER + filenameWithTr + "_fold" + str(foldId) + "_trainData" + ".npy")
        trainLabels = numpy.load(constants.BASE_FOLDER + filenameWithTr + "_fold" + str(foldId) + "_trainLabels" + ".npy")
        testData = numpy.load(constants.BASE_FOLDER + filenameWithTr + "_fold" + str(foldId) + "_testData" + ".npy")
        testLabels = numpy.load(constants.BASE_FOLDER + filenameWithTr + "_fold" + str(foldId) + "_testLabels" + ".npy")
        unlabeledData = numpy.load(constants.BASE_FOLDER + filenameWithTr + "_fold" + str(foldId) + "_unlabeledData" + ".npy")
        
        
    # use this to include interaction terms
    # trainData = addInteractionTerms(trainData)
    # unlabeledData = addInteractionTerms(unlabeledData)
    # testData = addInteractionTerms(testData)
    
    # print("nan in training data = ", numpy.count_nonzero(numpy.isnan(trainData)))
    # print("nan in test data = ", numpy.count_nonzero(numpy.isnan(testData)))
    
    # standardize data
    trainData, unlabeledData, testData = preprocessing.standardizeAllDataNew(trainData, unlabeledData, testData)

    # print("nan in training data = ", numpy.count_nonzero(numpy.isnan(trainData)))
    # print("nan in test data = ", numpy.count_nonzero(numpy.isnan(testData)))
    
    # test Data should not have any NAN entries 
    assert(not numpy.any(numpy.isnan(testData)))
    
    return trainData, trainLabels, unlabeledData, testData, testLabels



def loadSubset(dataName, trainSize, foldId, imputationMethod):
    trainData, trainLabels, unlabeledData, testData, testLabels = loadSubsetBasic(dataName, None, foldId)
        
    if dataName.endswith("WithMissing_5foldCV"):
        assert(imputationMethod is not None)
        trainData = numpy.load(constants.BASE_FOLDER + dataName + "_" + imputationMethod + "_fold" + str(foldId) + "_trainData" + ".npy")
        
    
    assert(not numpy.any(numpy.isnan(trainData)))
    assert(not numpy.any(numpy.isnan(unlabeledData)))
    assert(not numpy.any(numpy.isnan(testData)))
    
    trainLabels = experimentHelper.getLabelsStartingAtZero(trainLabels)
    testLabels = experimentHelper.getLabelsStartingAtZero(testLabels)
    
    return trainData, trainLabels, unlabeledData, testData, testLabels


def getAvgTrueLabelRatioOnTestData(dataName):
    
    trueRatio = 0.0
    for foldId in range(constants.NUMBER_OF_FOLDS):
        trainData, trainLabels, unlabeledData, testData, testLabels = loadSubset(dataName, None, foldId, imputationMethod = constants.IMPUTATION_METHOD)
        trueRatio += numpy.sum(testLabels) / testLabels.shape[0]

    avgTLR = trueRatio / constants.NUMBER_OF_FOLDS
    assert(avgTLR > 0.0 and avgTLR < 1.0)
    return avgTLR


# assume binary labels 1.0 (positive) and 0.0 (negative)
def getPositiveRatio(labels):
    assert(numpy.all(numpy.logical_or(labels == -1.0, labels == 1.0)))
    assert(len(labels.shape) == 1)
    return numpy.sum(labels == 1.0) / float(labels.shape[0]) 
    

def getLabelRatios(trainLabels, testLabels):
    print("positive label ratio training data = ", getPositiveRatio(trainLabels))
    print("positive label ratio test data = ", getPositiveRatio(testLabels))
    return
    


def prepareMiniBooNEData(trainSize):
    assert(trainSize == 100 or trainSize == 500 or trainSize == 1000 or trainSize is None)  
    
    testSize = 1000
    unlabeledSize = 10000
    
    if trainSize is not None:
        stemFilename = "miniBooNE_" + str(trainSize) + "tr" 
    else:
        trainSize = 500
        stemFilename = "miniBooNE_5foldCV"
    
    numpy.random.seed(3523421)
    
    data, labels, costs = loadMiniBooNEData()
    allTrainData, allTrainLabels, allUnlabeledData, allTestData, allTestLabels = getRandomSplitsNew(data, labels, trainSize, testSize, unlabeledSize)
    saveAllSubsetsNew(allTrainData, allTrainLabels, allUnlabeledData, allTestData, allTestLabels, stemFilename)
    
    print("FINISHED")


def prepareMiniBooNEDataForBiasEval(trainSize):
    assert(trainSize == 50 or trainSize == 200 or trainSize == 500)  
    
    NUMBER_OF_RANDOM_SPLITS = 10
    
    testSize = 10000
    unlabeledSize = 10000
    
    stemFilename = "miniBooNE_ForBiasEval_" + str(trainSize) + "tr" 
    
    numpy.random.seed(3523421)
    
    data, labels, costs = loadMiniBooNEData()
    allTrainData, allTrainLabels, allUnlabeledData, allTestData, allTestLabels = getRandomSplitsNew(data, labels, trainSize, testSize, unlabeledSize, NUMBER_OF_RANDOM_SPLITS)
    saveAllSubsetsNew(allTrainData, allTrainLabels, allUnlabeledData, allTestData, allTestLabels, stemFilename)
    
    print("FINISHED")


def preparePimaData(trainSize):
    assert(trainSize is None or trainSize == 100 or trainSize == 200)
    
    numpy.random.seed(3523421)   
    data, labels, costs, variableNames = loadPimaDiabetesData()
    
    if trainSize is None:
        stemFilename = "pima_5foldCV" 
        save5FoldCV_noUnlabeledData(data, labels, stemFilename)
    else:
        testSize = 200
        unlabeledSize = "remaining"
        stemFilename = "pima_" + str(trainSize) + "tr" 
        allTrainData, allTrainLabels, allUnlabeledData, allTestData, allTestLabels = getRandomSplitsNew(data, labels, trainSize, testSize, unlabeledSize)
        saveAllSubsetsNew(allTrainData, allTrainLabels, allUnlabeledData, allTestData, allTestLabels, stemFilename)
        
    print("FINISHED")

        

def prepareBreastCancerData(trainSize):
    assert(trainSize is None or trainSize == 100 or trainSize == 200)
    
    numpy.random.seed(3523421)
    
    data, labels, costs, variableNames = loadWisconsinBreastCancer()
    
    if trainSize is None:
        stemFilename = "breastcancer_5foldCV" 
        save5FoldCV_noUnlabeledData(data, labels, stemFilename)
    else:
        testSize = 200
        unlabeledSize = "remaining"
        stemFilename = "breastcancer_" + str(trainSize) + "tr" 
        allTrainData, allTrainLabels, allUnlabeledData, allTestData, allTestLabels = getRandomSplitsNew(data, labels, trainSize, testSize, unlabeledSize)
        saveAllSubsetsNew(allTrainData, allTrainLabels, allUnlabeledData, allTestData, allTestLabels, stemFilename)
        
    print("FINISHED")



def preparePyhsioNetDataNoMissingAttributes():
    
    numpy.random.seed(3523421)
    
    data = numpy.load(constants.BASE_FOLDER + physioNet.filenameData + ".npy")
    labels = numpy.load(constants.BASE_FOLDER + physioNet.filenameLabels + ".npy")

    # change to -1 and 1 labels
    labels[labels == 0] = -1
    
    sortedIds = numpy.sort(list(physioNet.VARIABLE_SUBSET_FOR_EVALUATION_NO_MISSING))
    
    # get filtered data
    validSampleIds = []
    for i in range(data.shape[0]):
        if not numpy.any(numpy.isnan(data[i,sortedIds])):
            validSampleIds.append(i)
    
    noMissingData = data[validSampleIds, :]
    noMissingData = noMissingData[:,sortedIds]
    
    noMissingDataLabels = labels[validSampleIds]
    
    print("total number of samples = ", noMissingData.shape[0])
    print("total number of variables = ", noMissingData.shape[1])
    assert(not numpy.any(numpy.isnan(noMissingData)))
    
    stemFilename = "pyhsioNetNoMissing_5foldCV" 
    save5FoldCV_noUnlabeledData(noMissingData, noMissingDataLabels, stemFilename)
       
    print("FINISHED")
    return



def prepareDataWithMissingAttributes(inputDataFilename, inputLabels, sortedVariableIds, outputStemFilename):
    
    numpy.random.seed(3523421)
    
    data = numpy.load(constants.BASE_FOLDER + inputDataFilename + ".npy")
    labels = numpy.load(constants.BASE_FOLDER + inputLabels + ".npy")

    # change to -1 and 1 labels
    labels[labels == 0] = -1
    
    # get all(!) samples with selected variables
    allData = data[:, sortedVariableIds]
    
    sampleIdsWithFullObservations = []
    for i in range(allData.shape[0]):
        sample = allData[i]
        if not numpy.any(numpy.isnan(sample)):
            sampleIdsWithFullObservations.append(i)
    
    sampleIdsWithFullObservations = numpy.asarray(sampleIdsWithFullObservations)
    sampleIdsWithMissingObservations = numpy.delete(numpy.arange(allData.shape[0]),  sampleIdsWithFullObservations)
    
    print("sampleIdsWithFullObservations = ", sampleIdsWithFullObservations)
    print("sampleIdsWithMissingObservations = ", sampleIdsWithMissingObservations)
    
    noMissingData = allData[sampleIdsWithFullObservations, :]
    noMissingData_labels = labels[sampleIdsWithFullObservations]
    withMissingData = allData[sampleIdsWithMissingObservations, :]
    withMissingData_labels = labels[sampleIdsWithMissingObservations]
    
    assert(noMissingData_labels.shape[0] + withMissingData_labels.shape[0] == allData.shape[0])
    
    print("noMissingData = ", noMissingData.shape)
    print("withMissingData = ", withMissingData.shape)
    
    filenameStemWithPath = constants.BASE_FOLDER + outputStemFilename
    
    NUMBER_OF_FOLDS = 5
    
    cvNoMissingData = sklearn.model_selection.StratifiedKFold(n_splits=NUMBER_OF_FOLDS, shuffle=True, random_state=564343)
    
    train_index_noMissing = []
    test_index_noMissing = []
    for train_index, test_index in cvNoMissingData.split(noMissingData, noMissingData_labels):
        train_index_noMissing.append(train_index)
        test_index_noMissing.append(test_index)

    print("all folds = ")
    for i in range(NUMBER_OF_FOLDS):
        trainData = numpy.vstack((noMissingData[train_index_noMissing[i]], withMissingData)) 
        trainLabels =  numpy.hstack((noMissingData_labels[train_index_noMissing[i]], withMissingData_labels))
        
        testData = noMissingData[test_index_noMissing[i]] 
        testLabels =  noMissingData_labels[test_index_noMissing[i]]
        
        assert(not numpy.any(numpy.isnan(testData)))
        assert(trainData.shape[0] + testData.shape[0] == allData.shape[0])
        assert(trainLabels.shape[0] + testLabels.shape[0] == allData.shape[0])
        
        print("fold = ", i)
        print("number of training samples (with no and missing data samples) = ", trainData.shape[0])
        print("number of test samples (all with no missing data) = ", testData.shape[0])
        
        numpy.save(filenameStemWithPath + "_fold" + str(i) + "_trainData", trainData)
        numpy.save(filenameStemWithPath + "_fold" + str(i) + "_trainLabels", numpy.asarray(trainLabels, dtype = numpy.int))
        numpy.save(filenameStemWithPath + "_fold" + str(i) + "_testData", testData)
        numpy.save(filenameStemWithPath + "_fold" + str(i) + "_testLabels", numpy.asarray(testLabels, dtype = numpy.int))
        unlabeledData = numpy.zeros((0, allData.shape[1]))
        numpy.save(filenameStemWithPath + "_fold" + str(i) + "_unlabeledData", unlabeledData)
         
     
    print("FINISHED")
    return



def preparePyhsioNetDataWithMissingAttributes():
    inputDataFilename = physioNet.filenameData
    inputLabels = physioNet.filenameLabels
    outputStemFilename = "pyhsioNetWithMissing_5foldCV"
    sortedVariableIds = numpy.sort(list(physioNet.VARIABLE_SUBSET_FOR_EVALUATION_WITH_MISSING))
    prepareDataWithMissingAttributes(inputDataFilename, inputLabels, sortedVariableIds, outputStemFilename)
    

def prepareHeartDiseaseDataWithMissingAttributes():
    inputDataFilename = "heartDisease_ClevelandData"
    inputLabels = "heartDisease_ClevelandLabels"
    outputStemFilename = "heartDiseaseWithMissing_5foldCV"
    sortedVariableIds = numpy.arange(13)
    prepareDataWithMissingAttributes(inputDataFilename, inputLabels, sortedVariableIds, outputStemFilename)


# prepareHeartDiseaseDataWithMissingAttributes()


# preparePyhsioNetDataWithMissingAttributes()

# preparePyhsioNetDataNoMissingAttributes()
# prepareCrabData()
# prepareMiniBooNEDataForBiasEval(50)

# prepareMiniBooNEData(None)
# prepareBreastCancerData(None)
# preparePimaData(None)
# dataName = "pima_5foldCV"
# dataName = "breastcancer_5foldCV"
# dataName = "miniBooNE_5foldCV"
# trainData, trainLabels, unlabeledData, testData, testLabels = loadSubsetNew(dataName, None, 0)
# print("p = ", trainData.shape[1])
# print("n = ", (trainData.shape[0] + testData.shape[0]))
# print("train = ", trainData.shape[0])
# print("test = ", testData.shape[0])
# print("unlabeled = ", unlabeledData.shape[0])

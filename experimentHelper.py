import numpy


# probs = probability p(y = 1) for each test sample
def getEstimatedAccuracy(probs):
    assert(len(probs.shape) == 1)
    
    classificationCorrectProbs = numpy.copy(probs)
    classificationCorrectProbs[probs < 0.5] = 1.0 - classificationCorrectProbs[probs < 0.5]
     
    return numpy.average(classificationCorrectProbs)


# probs = numpy.asarray([0.3, 0.7, 0.7, 0.6, 0.2, 0.99])
# estimatedTestAcc = getEstimatedAccuracy(probs)
# print("estimatedTestAcc = ", estimatedTestAcc)

    
def getLabelsStartingAtZero(labels):
    assert(numpy.all(numpy.logical_or(labels == -1, labels == 1)))
    labelsStartingAtZero = (labels + 1) / 2
    return labelsStartingAtZero.astype(numpy.int)
    

def showVector(vec):
    # print " ".join(vec)
    
    roundedElems = [str(round(elem, 2)) for elem in vec]
    print("\t".join(roundedElems)) 
    return

def showMatrix(A):
    print("[")
    for i in range(A.shape[0]):
        showVector(A[i])
    print("]")
    return


class ResultsRecorder:
    def __init__(self, numberOfSteps):
        
        # first column: misclassification costs
        # second column: average value
        # third column: standard deviation
        self.allTotalCosts = numpy.zeros((numberOfSteps, 3))
        self.allFeatureCosts = numpy.zeros((numberOfSteps, 3))
        self.allMisClassificationCosts = numpy.zeros((numberOfSteps, 3))
        self.allAccuracies = numpy.zeros((numberOfSteps, 3))
        self.allAUC = numpy.zeros((numberOfSteps, 3))
        
        
        self.allRecall = numpy.zeros((numberOfSteps, 3))
        self.allFDR = numpy.zeros((numberOfSteps, 3))
        self.allOperationCosts = numpy.zeros((numberOfSteps, 3))
        
        self.allRecall_atExactRecall = numpy.zeros((numberOfSteps, 3))
        self.allFDR_atExactRecall = numpy.zeros((numberOfSteps, 3))
        self.allOperationCosts_atExactRecall = numpy.zeros((numberOfSteps, 3))
        
        self.currentStepId = 0
        self.numberOfSteps = numberOfSteps
    
    
    def addAll(self, misclassificationCostsSymmetric, allResultsForThisStep):
         
        testTotalCostsAllFolds, testFeatureCostsAllFolds, testMisClassificationCostsAllFolds, testAccuracyAllFolds, testAUCAllFolds, testRecallAllFolds, testFDRAllFolds, testOperationCostAllFolds, testRecallAllFolds_atExactRecall , testFDRAllFolds_atExactRecall , testOperationCostAllFolds_atExactRecall = allResultsForThisStep
        
        self.allTotalCosts[self.currentStepId] = misclassificationCostsSymmetric, numpy.average(testTotalCostsAllFolds), numpy.std(testTotalCostsAllFolds)
        self.allFeatureCosts[self.currentStepId] = misclassificationCostsSymmetric, numpy.average(testFeatureCostsAllFolds), numpy.std(testFeatureCostsAllFolds)
        self.allMisClassificationCosts[self.currentStepId] = misclassificationCostsSymmetric, numpy.average(testMisClassificationCostsAllFolds), numpy.std(testMisClassificationCostsAllFolds)
        self.allAccuracies[self.currentStepId] = misclassificationCostsSymmetric, numpy.average(testAccuracyAllFolds), numpy.std(testAccuracyAllFolds)  
        self.allAUC[self.currentStepId] = misclassificationCostsSymmetric, numpy.average(testAUCAllFolds), numpy.std(testAUCAllFolds)
        
        # allRecall, allFDR, allOperationCosts
        self.allRecall[self.currentStepId] = misclassificationCostsSymmetric, numpy.average(testRecallAllFolds), numpy.std(testRecallAllFolds)
        self.allFDR[self.currentStepId] = misclassificationCostsSymmetric, numpy.average(testFDRAllFolds), numpy.std(testFDRAllFolds)
        self.allOperationCosts[self.currentStepId] = misclassificationCostsSymmetric, numpy.average(testOperationCostAllFolds), numpy.std(testOperationCostAllFolds)
        
        # allRecall, allFDR, allOperationCosts AT EXACT RECALL
        self.allRecall_atExactRecall[self.currentStepId] = misclassificationCostsSymmetric, numpy.average(testRecallAllFolds_atExactRecall), numpy.std(testRecallAllFolds_atExactRecall)
        self.allFDR_atExactRecall[self.currentStepId] = misclassificationCostsSymmetric, numpy.average(testFDRAllFolds_atExactRecall), numpy.std(testFDRAllFolds_atExactRecall)
        self.allOperationCosts_atExactRecall[self.currentStepId] = misclassificationCostsSymmetric, numpy.average(testOperationCostAllFolds_atExactRecall), numpy.std(testOperationCostAllFolds_atExactRecall)
        
        self.currentStepId += 1
        return
    
    
    BASE_FOLDER = "/export/home/s-andrade/newStart/eclipseWorkspaceDynamic/DynamicCovariateSelection/results/"
        
    
    def writeOutResults(self, namePrefix):
        assert(self.currentStepId == self.numberOfSteps)
        
        # print("SAVE TO " + ResultsRecorder.BASE_FOLDER + namePrefix + "_allTotalCosts")
        
        numpy.save(ResultsRecorder.BASE_FOLDER + namePrefix + "_allTotalCosts", self.allTotalCosts)
        numpy.save(ResultsRecorder.BASE_FOLDER + namePrefix + "_allFeatureCosts", self.allFeatureCosts)
        numpy.save(ResultsRecorder.BASE_FOLDER + namePrefix + "_allMisClassificationCosts", self.allMisClassificationCosts)
        numpy.save(ResultsRecorder.BASE_FOLDER + namePrefix + "_allAccuracies", self.allAccuracies)
        numpy.save(ResultsRecorder.BASE_FOLDER + namePrefix + "_allAUC", self.allAUC)
        
        numpy.save(ResultsRecorder.BASE_FOLDER + namePrefix + "_allRecall", self.allRecall)
        numpy.save(ResultsRecorder.BASE_FOLDER + namePrefix + "_allFDR", self.allFDR)
        numpy.save(ResultsRecorder.BASE_FOLDER + namePrefix + "_allOperationCosts", self.allOperationCosts)
        
        numpy.save(ResultsRecorder.BASE_FOLDER + namePrefix + "_allRecall_atExactRecall", self.allRecall_atExactRecall)
        numpy.save(ResultsRecorder.BASE_FOLDER + namePrefix + "_allFDR_atExactRecall", self.allFDR_atExactRecall)
        numpy.save(ResultsRecorder.BASE_FOLDER + namePrefix + "_allOperationCosts_atExactRecall", self.allOperationCosts_atExactRecall)
        
        return

    
    # each returned object is numpy array of shape (number of different misclassification costs, 3), where
    # column 0: misclassification costs
    # column 1: average 
    # column 2: standard deviation 
    @staticmethod
    def readResults(namePrefix):
        allTotalCosts = numpy.load(ResultsRecorder.BASE_FOLDER + namePrefix + "_allTotalCosts" + ".npy")
        allFeatureCosts = numpy.load(ResultsRecorder.BASE_FOLDER + namePrefix + "_allFeatureCosts" + ".npy")
        allMisClassificationCosts = numpy.load(ResultsRecorder.BASE_FOLDER + namePrefix + "_allMisClassificationCosts" + ".npy")
        allAccuracies = numpy.load(ResultsRecorder.BASE_FOLDER + namePrefix + "_allAccuracies" + ".npy")
        allAUC = numpy.load(ResultsRecorder.BASE_FOLDER + namePrefix + "_allAUC" + ".npy")
        
        allRecall = numpy.load(ResultsRecorder.BASE_FOLDER + namePrefix + "_allRecall" + ".npy")
        allFDR = numpy.load(ResultsRecorder.BASE_FOLDER + namePrefix + "_allFDR" + ".npy")
        allOperationCosts = numpy.load(ResultsRecorder.BASE_FOLDER + namePrefix + "_allOperationCosts" + ".npy")
        
        
        allRecall_atExactRecall = numpy.load(ResultsRecorder.BASE_FOLDER + namePrefix + "_allRecall_atExactRecall" + ".npy")
        allFDR_atExactRecall = numpy.load(ResultsRecorder.BASE_FOLDER + namePrefix + "_allFDR_atExactRecall" + ".npy")
        allOperationCosts_atExactRecall = numpy.load(ResultsRecorder.BASE_FOLDER + namePrefix + "_allOperationCosts_atExactRecall" + ".npy")
        
        return allTotalCosts, allFeatureCosts, allMisClassificationCosts, allAccuracies, allAUC, allRecall, allFDR, allOperationCosts, allRecall_atExactRecall, allFDR_atExactRecall, allOperationCosts_atExactRecall


# def getPredictionProbSingleObservation(predictionModel, observedCovariates):
#     observedCovariatesForClassifier = numpy.reshape(observedCovariates, (1, -1))  # classifier expects a 2D array
#     return predictionModel.predict_proba(observedCovariatesForClassifier)


# def getPredictionLabelSingleObservation(predictionModel, observedCovariates):
#     observedCovariatesForClassifier = numpy.reshape(observedCovariates, (1, -1))  # classifier expects a 2D array
#     return (predictionModel.predict(observedCovariatesForClassifier))[0]
        

def showAvgAndStd(resultArray, roundDigits = 2):
    assert(resultArray.shape[0] == 3)
    return str(round(resultArray[1], roundDigits)) + " (" + str(round(resultArray[2], roundDigits)) + ") " 

def showAvgAndStdInPercent(resultArray):
    assert(resultArray.shape[0] == 3)
    return str(round(resultArray[1] * 100,4)) + " (" + str(round(resultArray[2] * 100,4)) + ") " 

def createTotalCostsTable(dataName, allMethodNames, allLabels, allMisclassificationIds):
    allTotalCosts, allFeatureCosts, allMisClassificationCosts, allAccuracies, allAUC = ResultsRecorder.readResults(dataName + "_" + allMethodNames[0])
    resultStr = ""
    for mcId in allMisclassificationIds:
        resultStr += " & \\bf " + str(allTotalCosts[mcId, 0])
    resultStr += " \\\\"
    print(resultStr)
    print("\\midrule")

    for i in range(len(allMethodNames)):
        allTotalCosts, allFeatureCosts, allMisClassificationCosts, allAccuracies, allAUC = ResultsRecorder.readResults(dataName + "_" + allMethodNames[i])
        resultStr = "\\bf " + allLabels[i]
        for mcId in allMisclassificationIds:
            resultStr += " & " + showAvgAndStd(allTotalCosts[mcId])
            # resultStr += " & " + showAvgAndStd(allFeatureCosts[mcId])
#             # print("************ Misclassification Costs " + str(allTotalCosts[mcId, 0]) + "************")
#             # print(allLabels[i] + " = " + showAvgAndStd(allTotalCosts[mcId]))
            
        resultStr += " \\\\"
        print(resultStr)
        
    return


def createFeatureCostsTable(dataName, allMethodNames, allLabels, allMisclassificationIds):
    _, allFeatureCosts, _, _, _ = ResultsRecorder.readResults(dataName + "_" + allMethodNames[0])
    resultStr = ""
    for mcId in allMisclassificationIds:
        resultStr += " & \\bf " + str(allFeatureCosts[mcId, 0])
    resultStr += " \\\\"
    print(resultStr)
    print("\\midrule")

    for i in range(len(allMethodNames)):
        _, allFeatureCosts, _, _ , _= ResultsRecorder.readResults(dataName + "_" + allMethodNames[i])
        resultStr = "\\bf " + allLabels[i]
        for mcId in allMisclassificationIds:
            resultStr += " & " + showAvgAndStd(allFeatureCosts[mcId])
           
        resultStr += " \\\\"
        print(resultStr)
        
    return


def createAccuraciesTable(dataName, allMethodNames, allLabels, allMisclassificationIds):
    _, _, _, allAccuracies, allAUC = ResultsRecorder.readResults(dataName + "_" + allMethodNames[0])
    resultStr = ""
    for mcId in allMisclassificationIds:
        resultStr += " & \\bf " + str(allAccuracies[mcId, 0])
    resultStr += " \\\\"
    print(resultStr)
    print("\\midrule")

    for i in range(len(allMethodNames)):
        _, _, _, allAccuracies, allAUC  = ResultsRecorder.readResults(dataName + "_" + allMethodNames[i])
        resultStr = "\\bf " + allLabels[i]
        for mcId in allMisclassificationIds:
            resultStr += " & " + showAvgAndStd(allAccuracies[mcId], 3)
           
        resultStr += " \\\\"
        print(resultStr)
        
    return


# "predict" and "predict_proba" follows the output format of scipy classifier 
class ClassRatioClassifier:
    def __init__(self, trainLabels):
        assert(numpy.all(numpy.logical_or(trainLabels == 0, trainLabels == 1)))
        positiveProb = numpy.sum(trainLabels) / float(trainLabels.shape[0])
        self.classProbs = numpy.asarray([1.0 - positiveProb, positiveProb])
        self.classProbs = numpy.reshape(self.classProbs, (1,-1))
    
    def predict(self, dataMatrix):
        assert(len(dataMatrix.shape) == 2)
        assert(dataMatrix.shape[0] >= 1)
        predictedClass = numpy.asarray([numpy.argmax(self.classProbs)])
        return numpy.repeat(predictedClass, dataMatrix.shape[0])
    
    def predict_proba(self, dataMatrix):
        assert(len(dataMatrix.shape) == 2)
        assert(dataMatrix.shape[0] >= 1)
        return numpy.tile(self.classProbs, (dataMatrix.shape[0], 1)) 
    

# "predict" and "predict_proba" follows the output format of scipy regressors/classifiers 
class BaseRegressionModel:
    def __init__(self, baseMean, baseVariance):
#         assert(len(baseVariance.shape) == 0) # baseVariance needs to be a scalar
        self.baseMean = baseMean
        self.baseSTD = numpy.sqrt(baseVariance)
    
    def predict(self, dataMatrix, return_std):
        assert(len(dataMatrix.shape) == 2)
        assert(dataMatrix.shape[0] >= 1)
        assert(return_std == True)
        allMeans = numpy.asarray([self.baseMean])
        allSTDs = numpy.asarray([self.baseSTD])
        return numpy.repeat(allMeans, dataMatrix.shape[0]), numpy.repeat(allSTDs, dataMatrix.shape[0])
    

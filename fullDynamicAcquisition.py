import numpy
import prepareFeatureSets
import dynamicAcquisition
import experimentHelper

# 
# allDataOrdered = numpy.random.normal(size = (10, 4))
# covMatrix = numpy.cov(allDataOrdered.transpose(), bias=True)
# 
# print(covMatrix)
# 
# print("")
# newOrder = [2,3,0,1]
# covMatrix = covMatrix[:, newOrder]
# covMatrix = covMatrix[newOrder, :]
# 
# print(covMatrix)
# 
# def getConditionalMeanAndVariance(allData, idsGivenCovariates, idsQueryCovariates):
#     p = allData.shape[1]
#     idsMarginalizeOutCovariates = numpy.arange(p)
#     idsMarginalizeOutCovariates = numpy.delete(idsMarginalizeOutCovariates, numpy.hstack((idsQueryCovariates, idsGivenCovariates)))
#     assert(len(set(idsMarginalizeOutCovariates) | set(idsQueryCovariates) | set(idsGivenCovariates)) == p)
#            
#     # print("idsMarginalizeOutCovariates = ", idsMarginalizeOutCovariates)
#     # print("idsGivenCovariates = ", idsGivenCovariates)
#     # print("idsQueryCovariates = ", idsQueryCovariates)
#         
#     orderedIds = numpy.hstack((idsQueryCovariates, idsMarginalizeOutCovariates, idsGivenCovariates))
#     assert(orderedIds.shape[0] == p)
#     
#     allDataOrdered = allData[:, orderedIds]
#     covMatrix = numpy.cov(allDataOrdered.transpose(), bias=True)
#     
#     upperLeftBlockSize = idsQueryCovariates.shape[0] + idsMarginalizeOutCovariates.shape[0]
#     
#     upperRightBlock = covMatrix[0:upperLeftBlockSize, upperLeftBlockSize:p]
#     upperLeftBlock = covMatrix[0:upperLeftBlockSize, 0:upperLeftBlockSize]
#     lowerRightBlock = covMatrix[upperLeftBlockSize:p, upperLeftBlockSize:p]
#     
#     # relevant matrices for the characterization of the normal distribution of rest variables given "idsGivenCovariates" variables
#     sigma12TimesSigma22Inv = numpy.matmul(upperRightBlock, numpy.linalg.inv(lowerRightBlock))
#     newCovMatrix = upperLeftBlock - numpy.matmul(sigma12TimesSigma22Inv, upperRightBlock.transpose())
#     
#     # experimentHelper.showMatrix(sigma12TimesSigma22Inv)
#     # experimentHelper.showMatrix(newCovMatrix)
#     
#     # get the variables from query only
#     q = idsQueryCovariates.shape[0]
#     sigma12TimesSigma22InvQueryPart = sigma12TimesSigma22Inv[0:q, :]
#     newCovMatrixQueryPart = newCovMatrix[0:q, 0:q]
#     
#     assert(sigma12TimesSigma22Inv.shape[1] == len(idsGivenCovariates))
#     return sigma12TimesSigma22InvQueryPart, newCovMatrixQueryPart







def getMeanAndVariance(beta, fullCovMatrix, observedCovariates, idsQueryCovariates, idsMarginalizeOutCovariates, idsGivenCovariates):
    assert(observedCovariates.shape[0] == idsGivenCovariates.shape[0])
    
    p = fullCovMatrix.shape[0]
    newOrder = numpy.hstack((idsQueryCovariates, idsMarginalizeOutCovariates, idsGivenCovariates))
    assert(newOrder.shape[0] == p)
    
    covMatrix = numpy.copy(fullCovMatrix)
    covMatrix = covMatrix[:, newOrder]
    covMatrix = covMatrix[newOrder, :]
    
    upperLeftBlockSize = idsQueryCovariates.shape[0] + idsMarginalizeOutCovariates.shape[0]
    
    upperRightBlock = covMatrix[0:upperLeftBlockSize, upperLeftBlockSize:p]
    upperLeftBlock = covMatrix[0:upperLeftBlockSize, 0:upperLeftBlockSize]
    lowerRightBlock = covMatrix[upperLeftBlockSize:p, upperLeftBlockSize:p]
    
    # relevant matrices for the characterization of the normal distribution of rest variables given "idsGivenCovariates" variables
    sigma12TimesSigma22Inv = numpy.matmul(upperRightBlock, numpy.linalg.inv(lowerRightBlock))
    newCovMatrix = upperLeftBlock - numpy.matmul(sigma12TimesSigma22Inv, upperRightBlock.transpose())
    
    # get the variables from query only
    q = idsQueryCovariates.shape[0]
    sigma12TimesSigma22InvQueryPart = sigma12TimesSigma22Inv[0:q, :]
    newCovMatrixQueryPart = newCovMatrix[0:q, 0:q]
    
    
    assert(sigma12TimesSigma22InvQueryPart.shape[1] == observedCovariates.shape[0] and sigma12TimesSigma22InvQueryPart.shape[0] == idsQueryCovariates.shape[0])
    meanQueryPart = numpy.matmul(sigma12TimesSigma22InvQueryPart, observedCovariates)
    
    assert(len(meanQueryPart.shape) == 1)
    assert(meanQueryPart.shape[0] > 0)
    assert(meanQueryPart.shape[0] == idsQueryCovariates.shape[0])
    
    assert(sigma12TimesSigma22InvQueryPart.shape[0] > 0)
    if sigma12TimesSigma22InvQueryPart.shape[1] == 0:
        assert(idsGivenCovariates.shape[0] == 0)
        # this case means no given covariates
        assert(numpy.all(meanQueryPart == 0))  # we assume that each covariate has zero mean
    assert(newCovMatrixQueryPart.shape[0] > 0 and newCovMatrixQueryPart.shape[1] > 0)
    
    queryBetaPart = beta[observedCovariates.shape[0]:beta.shape[0]]
    
    mean = numpy.dot(queryBetaPart, meanQueryPart)
    variance = queryBetaPart @ newCovMatrixQueryPart @ queryBetaPart.transpose()
    
    return mean, variance


# checked
def runLinearSequenceDynamicAcquisition(definedFeatureCosts, misclassificationCosts, allFeatureSetsInOrder, allSamplerInfos, allPredictionModels, testSample, onlyLookingOneStepAhead, fullCovMatrix):
    assert(len(testSample.shape) == 1)
    assert(testSample.shape[0] == definedFeatureCosts.shape[0])
    assert(len(allFeatureSetsInOrder) == len(allSamplerInfos))
    assert(len(allFeatureSetsInOrder) == len(allPredictionModels))
    
    # print("Analyze new test sample")
    
    numberOfFeatureSets = len(allFeatureSetsInOrder)
    
    for i in range(numberOfFeatureSets):
        idsGivenCovariates = allFeatureSetsInOrder[i]
        observedCovariates = testSample[idsGivenCovariates]
        currentFeatureCosts = numpy.sum(definedFeatureCosts[idsGivenCovariates])
        
        # print("idsGivenCovariates = ", idsGivenCovariates)
        
        # handle the special case that all ids are False, i.e. no covariates are selected
        # if observedCovariates.shape[0] == 0:
        #    assert(numpy.all(numpy.bitwise_not(idsGivenCovariates)))
        #    idsGivenCovariates = numpy.zeros((0), dtype=numpy.int)
        
        currentPredictionModel = allPredictionModels[i]
        
        observedCovariatesForClassifier = numpy.reshape(observedCovariates, (1, -1))  # classifier expects a 2D array
        predictedProbs = currentPredictionModel.predict_proba(observedCovariatesForClassifier)
        assert(predictedProbs.shape[0] == 1 and predictedProbs.shape[1] == misclassificationCosts.shape[0])
        currentBayesRisk = prepareFeatureSets.bayesRisk(predictedProbs, misclassificationCosts)
        
        # print("currentBayesRisk = ", currentBayesRisk)
        
        continueAskingForFeatures = False
        
        for j, sampleInfoTriple in enumerate(allSamplerInfos[i]):
            
            nextPredictionModel = allPredictionModels[i + j + 1]
            
            idsQueryCovariates, idsMarginalizeOutCovariates, idsGivenCovariates = sampleInfoTriple
            assert(numpy.all(numpy.equal(allFeatureSetsInOrder[i + j + 1], numpy.append(idsGivenCovariates, idsQueryCovariates))))
            assert(nextPredictionModel.coef_.shape[1] == idsGivenCovariates.shape[0] + idsQueryCovariates.shape[0])

            beta = nextPredictionModel.coef_[0]
            mean, variance = getMeanAndVariance(beta, fullCovMatrix, observedCovariates, idsQueryCovariates, idsMarginalizeOutCovariates, idsGivenCovariates)
                        
            expectedFutureBayesRisk = dynamicAcquisition.getExpectedBayesRiskEstimateProposed(misclassificationCosts, observedCovariates, mean, variance, nextPredictionModel)
            
            additionalFeatureCosts = numpy.sum(definedFeatureCosts[idsQueryCovariates])
           
            # print("queryCovariateIds = " + str(queryCovariateIds) + ", expectedFutureBayesRisk = " + str(expectedFutureBayesRisk))
            totalExpectedFutureCosts = expectedFutureBayesRisk + additionalFeatureCosts  
            
            
            if totalExpectedFutureCosts < currentBayesRisk:
                continueAskingForFeatures = True
                break
            
            if onlyLookingOneStepAhead:
                break 
        
        if not continueAskingForFeatures:
            # print("idsGivenCovariates = ", idsGivenCovariates)
            # assert(False)
            print("covariateIds = " + str(idsGivenCovariates) + "| costs of acquired covariates = " + str(currentFeatureCosts) + " | bayes risk = " + str(currentBayesRisk))
            predictedLabel = currentPredictionModel.predict(observedCovariatesForClassifier)
            assert(len(predictedLabel.shape) == 1 and predictedLabel.shape[0] == 1)
            return idsGivenCovariates, currentFeatureCosts, predictedLabel[0]
    
    assert(False)
    return None



# mc-checked
def getBayesRiskFromCurrentTestSample(trainData, trainLabels, testSample, misclassificationCosts, selectedFeatureIds):
    predictionModel, holdOutDataAccuracyEstimate = prepareFeatureSets.getOptimalTrainedModel(trainData, trainLabels, selectedFeatureIds)
    
    predictedProbs = predictionModel.predict_proba(testSample)
    print(predictedProbs)
    br = prepareFeatureSets.bayesRisk(predictedProbs, misclassificationCosts)
    print(br)
    assert(False)
    return prepareFeatureSets.bayesRisk(predictedProbs, misclassificationCosts)


# use this for the ordinary linear regression model as density model 
def prepareForLinearSequence(trainDataFullCovariates, trainLabels, allData, allFeatureListsInOrder):
    
    p = trainDataFullCovariates.shape[1]
    
    allPredictionModels = []
    allSamplerInfos = []
    numberOfFeatureSets = len(allFeatureListsInOrder)
        
    for i in range(numberOfFeatureSets):
        idsGivenCovariates = allFeatureListsInOrder[i]
        
        # print("******************** i = " + str(i) + " ***********************")
        
        # train classifier
        predictionModel, _ = prepareFeatureSets.getOptimalTrainedModel(trainDataFullCovariates, trainLabels, idsGivenCovariates)
                  
        samplerInfo = []
        for j in range(i + 1, numberOfFeatureSets):
            idsNextCovariates = allFeatureListsInOrder[j]
            assert((trainDataFullCovariates[:, idsNextCovariates]).shape[1] > 0)
            
            idsQueryCovariates = idsNextCovariates[idsGivenCovariates.shape[0] : idsNextCovariates.shape[0]]
            
            idsMarginalizeOutCovariates = numpy.arange(p)
            idsMarginalizeOutCovariates = numpy.delete(idsMarginalizeOutCovariates, numpy.hstack((idsQueryCovariates, idsGivenCovariates)))
            assert(len(set(idsMarginalizeOutCovariates) | set(idsQueryCovariates) | set(idsGivenCovariates)) == p)
            samplerInfo.append((idsQueryCovariates, idsMarginalizeOutCovariates, idsGivenCovariates))
     
            
        allSamplerInfos.append(samplerInfo)
        allPredictionModels.append(predictionModel)
    
    fullCovMatrix = numpy.cov(allData.transpose(), bias=True)
    
    return fullCovMatrix, allSamplerInfos, allPredictionModels



def getTotalExpectedFutureCosts(fullCovMatrix, nextPredictionModel, misclassificationCosts, definedFeatureCosts, idsQueryCovariates, idsMarginalizeOutCovariates, idsGivenCovariates, observedCovariates):
    assert(nextPredictionModel.coef_.shape[1] == idsGivenCovariates.shape[0] + idsQueryCovariates.shape[0])

    beta = nextPredictionModel.coef_[0]
    mean, variance = getMeanAndVariance(beta, fullCovMatrix, observedCovariates, idsQueryCovariates, idsMarginalizeOutCovariates, idsGivenCovariates)
                
    expectedFutureBayesRisk = dynamicAcquisition.getExpectedBayesRiskEstimateProposed(misclassificationCosts, observedCovariates, mean, variance, nextPredictionModel)
    
    additionalFeatureCosts = numpy.sum(definedFeatureCosts[idsQueryCovariates])
   
    # print("queryCovariateIds = " + str(queryCovariateIds) + ", expectedFutureBayesRisk = " + str(expectedFutureBayesRisk))
    totalExpectedFutureCosts = expectedFutureBayesRisk + additionalFeatureCosts  
    return totalExpectedFutureCosts



# CHECKED
def findNextBestCovariateId(fullCovMatrix, trainData, trainLabels, misclassificationCosts, definedFeatureCosts, idsGivenCovariates, observedCovariates, modelContainer, lock):
    assert(trainData.shape[0] == trainLabels.shape[0])
    p = trainData.shape[1]
    
    remainingCovariates = set(numpy.arange(p)) - set(idsGivenCovariates)
    
    fixedQueryCovariates = numpy.asarray([], dtype = numpy.int)
    
    globalBestTotalExpectedFutureCosts = float("inf")
    
    if len(remainingCovariates) == 0:
        # no new covariates can be acquired
        return None, float("inf")
    
    while len(remainingCovariates) > 0:
        bestLocalNextQueryId = None
        bestLocalTotalExpectedFutureCosts = float("inf")
        
        for j in remainingCovariates:
            idsQueryCovariates = numpy.append(fixedQueryCovariates, [j]) 
            covariatesForPrediction = numpy.append(idsGivenCovariates, idsQueryCovariates)
            
            idsMarginalizeOutCovariates = numpy.arange(p)
            idsMarginalizeOutCovariates = numpy.delete(idsMarginalizeOutCovariates, covariatesForPrediction)
            assert(len(set(idsMarginalizeOutCovariates) | set(idsQueryCovariates) | set(idsGivenCovariates)) == p)
            
            predictionModel, _ = modelContainer.getOptimalTrainedModel(covariatesForPrediction, lock) 
        
            totalExpectedFutureCosts = getTotalExpectedFutureCosts(fullCovMatrix, predictionModel, misclassificationCosts, definedFeatureCosts, idsQueryCovariates, idsMarginalizeOutCovariates, idsGivenCovariates, observedCovariates)
            if totalExpectedFutureCosts < bestLocalTotalExpectedFutureCosts: 
                bestLocalNextQueryId = j
                bestLocalTotalExpectedFutureCosts = totalExpectedFutureCosts 
        
        
        fixedQueryCovariates = numpy.append(fixedQueryCovariates, [bestLocalNextQueryId])
        remainingCovariates.remove(bestLocalNextQueryId)
        
        if bestLocalTotalExpectedFutureCosts < globalBestTotalExpectedFutureCosts:
            globalBestTotalExpectedFutureCosts = bestLocalTotalExpectedFutureCosts

    return fixedQueryCovariates[0], globalBestTotalExpectedFutureCosts


def findNextBestCovariateIdDEBUG(fullCovMatrix, trainData, trainLabels, misclassificationCosts, definedFeatureCosts, idsGivenCovariates, observedCovariates, allFeatureListsInOrder):
    assert(trainData.shape[0] == trainLabels.shape[0])
    p = trainData.shape[1]
    
    remainingCovariates = set(numpy.arange(p)) - set(idsGivenCovariates)
    fixedQueryCovariates = numpy.asarray([], dtype = numpy.int)
    
    if len(remainingCovariates) == 0:
        # no new covariates can be acquired
        return None, float("inf")
    
    numberOfFeatureSets = len(allFeatureListsInOrder)
    currentIdForGivenFeatures = None
    
    for i in range(numberOfFeatureSets):
        if len(idsGivenCovariates) == len(allFeatureListsInOrder[i]):
            assert(numpy.all(numpy.equal(idsGivenCovariates, allFeatureListsInOrder[i])))
            currentIdForGivenFeatures = i
            break
    
    assert(currentIdForGivenFeatures is not None)
    
    bestLocalTotalExpectedFutureCosts = float("inf")
                                              
    for j in range(currentIdForGivenFeatures + 1, numberOfFeatureSets):
        idsNextCovariates = allFeatureListsInOrder[j]
        idsQueryCovariates = idsNextCovariates[idsGivenCovariates.shape[0] : idsNextCovariates.shape[0]]

        covariatesForPrediction = numpy.append(idsGivenCovariates, idsQueryCovariates)
            
        idsMarginalizeOutCovariates = numpy.arange(p)
        idsMarginalizeOutCovariates = numpy.delete(idsMarginalizeOutCovariates, covariatesForPrediction)
        assert(len(set(idsMarginalizeOutCovariates) | set(idsQueryCovariates) | set(idsGivenCovariates)) == p)
        
        predictionModel, _ = prepareFeatureSets.getOptimalTrainedModel(trainData, trainLabels, covariatesForPrediction)
    
        totalExpectedFutureCosts = getTotalExpectedFutureCosts(fullCovMatrix, predictionModel, misclassificationCosts, definedFeatureCosts, idsQueryCovariates, idsMarginalizeOutCovariates, idsGivenCovariates, observedCovariates)
        if totalExpectedFutureCosts < bestLocalTotalExpectedFutureCosts: 
            bestLocalTotalExpectedFutureCosts = totalExpectedFutureCosts 

        fixedQueryCovariates = numpy.append(fixedQueryCovariates, idsQueryCovariates)
        
    return fixedQueryCovariates[0], bestLocalTotalExpectedFutureCosts


# CHECKED
def fullDynamicGreedyMethod(fullCovMatrix, trainData, trainLabels, testSample, misclassificationCosts, definedFeatureCosts, initialFeatureId, modelContainer, lock):
        
    idsGivenCovariates = numpy.asarray([initialFeatureId], dtype = numpy.int)
    
    while(True):
        
        observedCovariates = testSample[idsGivenCovariates]
        currentPredictionModel, _ = modelContainer.getOptimalTrainedModel(idsGivenCovariates, lock)
        observedCovariatesForClassifier = numpy.reshape(observedCovariates, (1, -1))  # classifier expects a 2D array
        predictedProbs = currentPredictionModel.predict_proba(observedCovariatesForClassifier)
        assert(predictedProbs.shape[0] == 1 and predictedProbs.shape[1] == misclassificationCosts.shape[0])
        currentBayesRisk = prepareFeatureSets.bayesRisk(predictedProbs, misclassificationCosts)
        
        bestNextQueryId, expectedTotalFutureCosts = findNextBestCovariateId(fullCovMatrix, trainData, trainLabels, misclassificationCosts, definedFeatureCosts, idsGivenCovariates, observedCovariates, modelContainer, lock)
        
        if expectedTotalFutureCosts < currentBayesRisk:
            # acquire covariate "bestNextQueryId"
            idsGivenCovariates = numpy.append(idsGivenCovariates, [bestNextQueryId])
        else:
            # decide on class label
            predictedLabel = currentPredictionModel.predict(observedCovariatesForClassifier)
            featureCosts = numpy.sum(definedFeatureCosts[idsGivenCovariates])
            print("covariateIds = " + str(idsGivenCovariates) + "| costs of acquired covariates = " + str(featureCosts) + " | bayes risk = " + str(currentBayesRisk))
            return idsGivenCovariates, featureCosts, predictedLabel[0]
    
    return






# ********************* FOR DEBUGGING **********************************************

# CHECKED
def fullDynamicGreedyMethodForDebugging(fullCovMatrix, trainData, trainLabels, testSample, misclassificationCosts, definedFeatureCosts, initialFeatureId):
    
    if initialFeatureId is None:
        idsGivenCovariates = numpy.asarray([], dtype = numpy.int)
    else:
        idsGivenCovariates = numpy.asarray([initialFeatureId], dtype = numpy.int)
    
    
    while(True):
        
        observedCovariates = testSample[idsGivenCovariates]
        currentPredictionModel, _ = prepareFeatureSets.getOptimalTrainedModel(trainData, trainLabels, idsGivenCovariates)
        currentBayesRisk = prepareFeatureSets.getBayesRiskSingleObservation(currentPredictionModel, misclassificationCosts, observedCovariates)
        
        bestNextQueryId, expectedTotalFutureCosts = findNextBestCovariateIdForDebugging(fullCovMatrix, trainData, trainLabels, misclassificationCosts, definedFeatureCosts, idsGivenCovariates, observedCovariates)
        
        if expectedTotalFutureCosts < currentBayesRisk:
            # acquire covariate "bestNextQueryId"
            idsGivenCovariates = numpy.append(idsGivenCovariates, [bestNextQueryId])
        else:
            # decide on class label
            predictedLabel = experimentHelper.getPredictionLabelSingleObservation(currentPredictionModel, observedCovariates)
            featureCosts = numpy.sum(definedFeatureCosts[idsGivenCovariates])
            print("covariateIds = " + str(idsGivenCovariates) + "| costs of acquired covariates = " + str(featureCosts) + " | bayes risk = " + str(currentBayesRisk))
            return idsGivenCovariates, featureCosts, predictedLabel
    
    return


# CHECKED
def fullDynamicGreedyMethodWithGoBack(fullCovMatrix, trainData, trainLabels, testSample, misclassificationCosts, definedFeatureCosts, initialFeatureId):
    
    if initialFeatureId is None:
        idsAllGivenCovariates = numpy.asarray([], dtype = numpy.int)
    else:
        idsAllGivenCovariates = numpy.asarray([initialFeatureId], dtype = numpy.int)
        
    
    lowestBayesRiskCovariateIds = numpy.asarray([], dtype = numpy.int)
    lowestBayesRiskClassifier, _ = prepareFeatureSets.getOptimalTrainedModel(trainData, trainLabels, lowestBayesRiskCovariateIds)
    lowestBayesRisk = prepareFeatureSets.getBayesRiskSingleObservation(lowestBayesRiskClassifier, misclassificationCosts, testSample[lowestBayesRiskCovariateIds])
    
    print("lowestBayesRisk = ", lowestBayesRisk)
    
    while(True):
        
        observedCovariates = testSample[idsAllGivenCovariates]
        currentPredictionModel, _ = prepareFeatureSets.getOptimalTrainedModel(trainData, trainLabels, idsAllGivenCovariates)
        currentBayesRisk = prepareFeatureSets.getBayesRiskSingleObservation(currentPredictionModel, misclassificationCosts, observedCovariates)
       
        print("idsGivenCovariates = " + str(idsAllGivenCovariates) + " | bayesRisk = " + str(currentBayesRisk))
       
        if currentBayesRisk < lowestBayesRisk:
            lowestBayesRisk = currentBayesRisk
            lowestBayesRiskClassifier = currentPredictionModel
            lowestBayesRiskCovariateIds = numpy.copy(idsAllGivenCovariates)
          
        bestNextQueryId, expectedTotalFutureCosts = findNextBestCovariateIdForDebugging(fullCovMatrix, trainData, trainLabels, misclassificationCosts, definedFeatureCosts, idsAllGivenCovariates, observedCovariates)
        
        if expectedTotalFutureCosts < lowestBayesRisk:
            # acquire covariate "bestNextQueryId"
            idsAllGivenCovariates = numpy.append(idsAllGivenCovariates, [bestNextQueryId])
        else:
#             lowestBayesRisk = float("inf")
#             lowestBayesRiskClassifier = None
#             lowestBayesRiskCovariateIds = None
#             
#             # decide on class label
#             print("NOW DECIDING ON CLASS LABEL")
#             print("idsAllGivenCovariates = ", idsAllGivenCovariates)
#             for j in range(len(idsAllGivenCovariates) + 1):
#                 idsGivenCovariates = idsAllGivenCovariates[0:j]
#                 observedCovariates = testSample[idsGivenCovariates]
#                 predictionModel, _ = prepareFeatureSets.getOptimalTrainedModel(trainData, trainLabels, idsGivenCovariates)
#                 bayesRisk = prepareFeatureSets.getBayesRiskSingleObservation(predictionModel, misclassificationCosts, observedCovariates)
#                 print("idsGivenCovariates = " + str(idsGivenCovariates) + " | bayesRisk = " + str(bayesRisk))
#                 if bayesRisk < lowestBayesRisk:
#                     lowestBayesRisk = bayesRisk
#                     lowestBayesRiskClassifier = predictionModel
#                     lowestBayesRiskCovariateIds = idsGivenCovariates
#             
#             if len(lowestBayesRiskCovariateIds) < len(idsAllGivenCovariates):
#                 print("!!!! BEST CLASSIFIER REQUIRES LESS COVARIATES THAN REQUIRED !!!! ")
            
            predictedLabel = experimentHelper.getPredictionLabelSingleObservation(lowestBayesRiskClassifier, testSample[lowestBayesRiskCovariateIds])
            featureCosts = numpy.sum(definedFeatureCosts[idsAllGivenCovariates])
            print("used covariate ids = " + str(lowestBayesRiskCovariateIds) + ", acquired covariateIds = " + str(lowestBayesRiskCovariateIds) + "| costs of acquired covariates = " + str(featureCosts) + " | bayes risk = " + str(lowestBayesRisk))
            # return the ids of the acquired(!) covariates, its costs, and the predicted label (which is based on lowestBayesRiskCovariateIds)
            return idsAllGivenCovariates, featureCosts, predictedLabel
    
    return


# CHECKED
def findNextBestCovariateIdForDebugging(fullCovMatrix, trainData, trainLabels, misclassificationCosts, definedFeatureCosts, idsGivenCovariates, observedCovariates):
    assert(trainData.shape[0] == trainLabels.shape[0])
    p = trainData.shape[1]
    
    remainingCovariates = set(numpy.arange(p)) - set(idsGivenCovariates)
    
    fixedQueryCovariates = numpy.asarray([], dtype = numpy.int)
    
    globalBestTotalExpectedFutureCosts = float("inf")
    
    if len(remainingCovariates) == 0:
        # no new covariates can be acquired
        return None, float("inf")
    
    while len(remainingCovariates) > 0:
        bestLocalNextQueryId = None
        bestLocalTotalExpectedFutureCosts = float("inf")
        
        for j in remainingCovariates:
            idsQueryCovariates = numpy.append(fixedQueryCovariates, [j]) 
            covariatesForPrediction = numpy.append(idsGivenCovariates, idsQueryCovariates)
            
            idsMarginalizeOutCovariates = numpy.arange(p)
            idsMarginalizeOutCovariates = numpy.delete(idsMarginalizeOutCovariates, covariatesForPrediction)
            assert(len(set(idsMarginalizeOutCovariates) | set(idsQueryCovariates) | set(idsGivenCovariates)) == p)
            
            predictionModel, _ = prepareFeatureSets.getOptimalTrainedModel(trainData, trainLabels, covariatesForPrediction)
            
            totalExpectedFutureCosts = getTotalExpectedFutureCosts(fullCovMatrix, predictionModel, misclassificationCosts, definedFeatureCosts, idsQueryCovariates, idsMarginalizeOutCovariates, idsGivenCovariates, observedCovariates)
            if totalExpectedFutureCosts < bestLocalTotalExpectedFutureCosts: 
                bestLocalNextQueryId = j
                bestLocalTotalExpectedFutureCosts = totalExpectedFutureCosts 
        
        
        fixedQueryCovariates = numpy.append(fixedQueryCovariates, [bestLocalNextQueryId])
        remainingCovariates.remove(bestLocalNextQueryId)
        
        if bestLocalTotalExpectedFutureCosts < globalBestTotalExpectedFutureCosts:
            globalBestTotalExpectedFutureCosts = bestLocalTotalExpectedFutureCosts

    return fixedQueryCovariates[0], globalBestTotalExpectedFutureCosts

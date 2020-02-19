import numpy
import sklearn.linear_model
import evaluation
import sklearn.metrics
from experimentHelper import ClassRatioClassifier
from multiprocessing import Pool
import time


def selectFeatureSets(definedFeatureCosts, misclassificationCosts, trainLabels, allFeatureArraysInOrder, allTrainingTrueProbsAllModels):
    
    allTotalCosts = []
                
    for i in range(len(allFeatureArraysInOrder)):
        idsGivenCovariates = allFeatureArraysInOrder[i]
        predictedTrueProbs = allTrainingTrueProbsAllModels[i]
        predictedLabels = evaluation.bayesClassifier(predictedTrueProbs, misclassificationCosts)
        totalCostEstimate = evaluation.getAverageMisclassificationCosts(trainLabels, predictedLabels, misclassificationCosts) + numpy.sum(definedFeatureCosts[idsGivenCovariates])
        allTotalCosts.append(totalCostEstimate)
    
    bestModelId = numpy.argmin(allTotalCosts)
    
    print("found covariate sets = ")
    for i in range(len(allFeatureArraysInOrder)):
        print("covariateIds = " + str(allFeatureArraysInOrder[i]) + " | expected total costs = " + str(allTotalCosts[i]))
     
    return allTotalCosts[bestModelId], bestModelId



def getAllFeatureSetsInOrderWithL1LogReg(trainDataWithoutCostScaling, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts):
    assert(numpy.all(numpy.logical_or(trainLabels == 0, trainLabels == 1)))
    assert(numpy.all(definedFeatureCosts > 0.0))
    
    # scale appropriately by inverse of feature costs
    trainData = trainDataWithoutCostScaling / definedFeatureCosts
    
    lambdaRegValues = numpy.arange(-10.0, 20.0, step = 0.5)
    lambdaRegValues = numpy.flip(lambdaRegValues, axis = 0)
    
    lambdaRegValues = numpy.append(lambdaRegValues, [-numpy.inf]) 
    
    allFeatureSetsInOrder = []
    allEstimatedTotalCosts = []
    
    for i, logAlphaValue in enumerate(lambdaRegValues):
        
        if logAlphaValue == -numpy.inf:
            selectedVariableIds = numpy.arange(trainData.shape[1])
        else:
            alphaValue = 2 ** logAlphaValue
            
            logReg = sklearn.linear_model.LogisticRegression(penalty="l1",C= 1.0 / alphaValue)
            logReg.fit(trainData, trainLabels)
            learnedBeta = logReg.coef_[0]
            
            selectedVariableIds = numpy.where(learnedBeta != 0)[0]
            
        notAlreadyFound = True
        for alreadyFoundFeatureSet in allFeatureSetsInOrder:
            if set(selectedVariableIds) == set(alreadyFoundFeatureSet):
                notAlreadyFound = False
                break
        
        if notAlreadyFound:
            allFeatureSetsInOrder.append(selectedVariableIds)
            if misclassificationCosts is not None:
                # totalCosts = getTotalCostsFromAllData(trainData, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts, selectedVariableIds)
                _, _, totalCosts = getOptimalTrainedModel_withTotalCosts(trainData, trainLabels, selectedVariableIds, misclassificationCosts, definedFeatureCosts, classificationModelName = "logReg", falsePositiveCost = None, targetRecall = None)
                allEstimatedTotalCosts.append(totalCosts)
            else:
                allEstimatedTotalCosts.append(None)
            
    print("found covariate sets = ")
    for i in range(len(allFeatureSetsInOrder)):
        print("covariateIds = " + str(allFeatureSetsInOrder[i]) + " | expected total costs = " + str(allEstimatedTotalCosts[i]))
        
    return allFeatureSetsInOrder, allEstimatedTotalCosts


    




def getBayesRiskSingleObservation(predictionModel, misclassificationCosts, observedCovariates):
    observedCovariatesForClassifier = numpy.reshape(observedCovariates, (1, -1))  # classifier expects a 2D array
    predictedProbs = predictionModel.predict_proba(observedCovariatesForClassifier)
    assert(predictedProbs.shape[0] == 1 and predictedProbs.shape[1] == misclassificationCosts.shape[0])
    currentBayesRisk = evaluation.bayesRisk(predictedProbs, misclassificationCosts)
    return currentBayesRisk


# mc-checked
# def getTotalCostsFromAllData(trainData, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts, selectedFeatureIds, classificationModelName, falsePositiveCost, targetRecall):
#     predictedProbsBestModel, predictionModel, totalCostEstimate = getOptimalTrainedModel_withMisclassificationCosts(trainData, trainLabels, selectedFeatureIds, misclassificationCosts, definedFeatureCosts, classificationModelName, falsePositiveCost, targetRecall)
#     
#     if unlabeledData.shape[0] == 0:
#         # assert(misclassificationCosts[0,1] == misclassificationCosts[1,0])
#         return totalCostEstimate
#     else:
#         assert(False) # NOT UPDATED
#         assert(unlabeledData.shape[0] > 0)
#         predictedProbs = predictionModel.predict_proba(unlabeledData[:,selectedFeatureIds])
#         assert(predictedProbs.shape[0] == unlabeledData.shape[0] and predictedProbs.shape[1] == misclassificationCosts.shape[0])
#         return evaluation.bayesRisk(predictedProbs, misclassificationCosts) + numpy.sum(definedFeatureCosts[selectedFeatureIds])
         


# def getFalseNegativeCostEstimate_forLowerBoundOnRecall(allThresholds, falsePositiveCost):
#     
#     minTau = numpy.min(allThresholds)
#     
#     assert(minTau > 0.0 and minTau < 1.0)
#     
#     return ((1.0/minTau) - 1.0) * falsePositiveCost
    

def getFalseNegativeCostEstimate_fromThresholdValue(minTau, falsePositiveCost):
    
    assert(minTau > 0.0 and minTau < 1.0)
    
    return ((1.0/minTau) - 1.0) * falsePositiveCost

    
# !NEW! updated
def getAllFeatureSetsInOrderWithGreedyMethod_normal(trainData, trainLabels, misclassificationCosts, definedFeatureCosts, classificationModelName, falsePositiveCost, targetRecall):
    
    allPredictedProbs = []
    allModels = []
    allFeatureSetsInOrder = []
    allEstimatedTotalCosts = []
    
    # add the empty feature set and classifier that does not use any features
    selectedFeatureIds = numpy.asarray([], dtype = numpy.int)
    predictedProbsInitialModel, initialModel, totalCosts = getOptimalTrainedModel_withTotalCosts(trainData, trainLabels, selectedFeatureIds, misclassificationCosts, definedFeatureCosts, classificationModelName, falsePositiveCost, targetRecall)
    
    allPredictedProbs.append(predictedProbsInitialModel)
    allModels.append(initialModel)
    allFeatureSetsInOrder.append(selectedFeatureIds)
    allEstimatedTotalCosts.append(totalCosts)
        
    
    # add the empty feature set and classifier that does not use any features
    # selectedFeatureIds = numpy.asarray([], dtype = numpy.int)
    # totalCosts = getTotalCostsFromAllData(trainData, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts, selectedFeatureIds, classificationModelName, falsePositiveCost, targetRecall)
    # allFeatureSetsInOrder.append(selectedFeatureIds)
    # allEstimatedTotalCosts.append(totalCosts)
    
    
    p = trainData.shape[1]
    currentSelectedFeatures = []
    notSelectedFeatureIds = set(numpy.arange(p))
    
    while len(notSelectedFeatureIds) > 0:
        
        bestCosts = float("inf")
        bestFeatureId = None
        predictedProbsBestModel = None
        bestModel = None
        
        # ORIGINAL PART
        for featureId in notSelectedFeatureIds:
            checkFeatureIds = list(currentSelectedFeatures)
            checkFeatureIds.append(featureId)
            checkFeatureIds = numpy.asarray(checkFeatureIds, dtype = numpy.int)
            assert(len(set(checkFeatureIds)) == len(set(currentSelectedFeatures)) + 1)
            assert(len(set(checkFeatureIds)) == len(checkFeatureIds)) 
            
            # bayesRisk = getBayesRiskFromAllData(trainData, trainLabels, unlabeledData, misclassificationCosts, checkFeatureIds)
            # totalFeatureCosts = numpy.sum(definedFeatureCosts[checkFeatureIds])
            # currentCosts = bayesRisk + totalFeatureCosts
            # currentCosts = getTotalCostsFromAllData(trainData, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts, checkFeatureIds, classificationModelName, falsePositiveCost, targetRecall)
            predictedProbsCurrentModel, currentModel, currentCosts = getOptimalTrainedModel_withTotalCosts(trainData, trainLabels, checkFeatureIds, misclassificationCosts, definedFeatureCosts, classificationModelName, falsePositiveCost, targetRecall)
    
            if currentCosts < bestCosts:
                bestCosts = currentCosts
                bestFeatureId = featureId
                bestModel = currentModel
                predictedProbsBestModel = predictedProbsCurrentModel
        
        assert(bestFeatureId is not None)
        currentSelectedFeatures.append(bestFeatureId)
        notSelectedFeatureIds.remove(bestFeatureId)
        
        allPredictedProbs.append(predictedProbsBestModel)
        allModels.append(bestModel)
        allFeatureSetsInOrder.append(numpy.asarray(currentSelectedFeatures, dtype = numpy.int))
        allEstimatedTotalCosts.append(bestCosts)

    print("found covariate sets = ")
    for i in range(len(allFeatureSetsInOrder)):
        print("covariateIds = " + str(allFeatureSetsInOrder[i]) + " | expected total costs = " + str(allEstimatedTotalCosts[i]))
    
    return allPredictedProbs, allModels, allFeatureSetsInOrder, allEstimatedTotalCosts


def getAllFeatureSetsInOrderWithGreedyMethod_normalWithoutCV(trainData, trainLabels):
    
    allFeatureSetsInOrder = []
    
    # add the empty feature set and classifier that does not use any features
    allFeatureSetsInOrder.append(numpy.asarray([], dtype = numpy.int))
    
    p = trainData.shape[1]
    currentSelectedFeatures = []
    notSelectedFeatureIds = set(numpy.arange(p))
    
    while len(notSelectedFeatureIds) > 0:
        
        bestCosts = float("inf")
        bestFeatureId = None
            
        # ORIGINAL PART
        for featureId in notSelectedFeatureIds:
            checkFeatureIds = list(currentSelectedFeatures)
            checkFeatureIds.append(featureId)
            checkFeatureIds = numpy.asarray(checkFeatureIds, dtype = numpy.int)
            assert(len(set(checkFeatureIds)) == len(set(currentSelectedFeatures)) + 1)
            assert(len(set(checkFeatureIds)) == len(checkFeatureIds)) 
            
            _, currentCosts = evaluation.getBestGAM_withoutCV(trainData, trainLabels, checkFeatureIds)
            if currentCosts < bestCosts:
                bestCosts = currentCosts
                bestFeatureId = featureId
        
        currentSelectedFeatures.append(bestFeatureId)
        notSelectedFeatureIds.remove(bestFeatureId)
        
        allFeatureSetsInOrder.append(numpy.asarray(currentSelectedFeatures, dtype = numpy.int))
        
    print("found covariate sets = ")
    for i in range(len(allFeatureSetsInOrder)):
        print("covariateIds = " + str(allFeatureSetsInOrder[i]))
    
    return allFeatureSetsInOrder



# !NEW! updated
def getAllFeatureSetsInOrderWithGreedyMethod_lazy(trainData, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts, classificationModelName, falsePositiveCost, targetRecall):
    
    allFeatureSetsInOrder = []
    allEstimatedTotalCosts = []
    
    # add the empty feature set and classifier that does not use any features
    selectedFeatureIds = numpy.asarray([], dtype = numpy.int)
    currentCosts = getTotalCostsFromAllData(trainData, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts, selectedFeatureIds, classificationModelName, falsePositiveCost, targetRecall)
    allFeatureSetsInOrder.append(selectedFeatureIds)
    allEstimatedTotalCosts.append(currentCosts)
    
    p = trainData.shape[1]
    currentSelectedFeatures = []
    notSelectedFeatureIds_set = set(numpy.arange(p))
        
    diminishingCostReductionUpperBounds = numpy.zeros(p)
    
    for featureId in notSelectedFeatureIds_set:
        checkFeatureIds = list(currentSelectedFeatures)
        checkFeatureIds.append(featureId)
        checkFeatureIds = numpy.asarray(checkFeatureIds, dtype = numpy.int)
        assert(len(set(checkFeatureIds)) == len(set(currentSelectedFeatures)) + 1)
        assert(len(set(checkFeatureIds)) == len(checkFeatureIds)) 
        
        newCosts = getTotalCostsFromAllData(trainData, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts, checkFeatureIds, classificationModelName, falsePositiveCost, targetRecall)
        diminishingCostReductionUpperBounds[featureId] = currentCosts - newCosts
    
    
    while len(notSelectedFeatureIds_set) > 0:
        
        bestTrueCostReduction = float("-inf")
        bestFeatureId = None
        
        notSelectedFeatureIds = numpy.asarray(list(notSelectedFeatureIds_set))
        notSelectedFeatureIds_sortIndex = numpy.argsort(- diminishingCostReductionUpperBounds[notSelectedFeatureIds])
        
        notSelectedFeatureIdsInOrder = notSelectedFeatureIds[notSelectedFeatureIds_sortIndex]
        costReductionUpperBoundsInOrder = diminishingCostReductionUpperBounds[notSelectedFeatureIdsInOrder]
        
        print("**************************")
        print("notSelectedFeatureIds = ", notSelectedFeatureIds)
        print("diminishingCostReductionBounds = ", diminishingCostReductionUpperBounds[notSelectedFeatureIds])
        print("notSelectedFeatureIdsInOrder = ", notSelectedFeatureIdsInOrder)
        print("costReductionUpperBoundsInOrder = ", costReductionUpperBoundsInOrder)
        # assert(False)
        
        for i, featureId in enumerate(notSelectedFeatureIdsInOrder):
            checkFeatureIds = list(currentSelectedFeatures)
            checkFeatureIds.append(featureId)
            checkFeatureIds = numpy.asarray(checkFeatureIds, dtype = numpy.int)
            assert(len(set(checkFeatureIds)) == len(set(currentSelectedFeatures)) + 1)
            assert(len(set(checkFeatureIds)) == len(checkFeatureIds)) 
            
            newCosts = getTotalCostsFromAllData(trainData, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts, checkFeatureIds, classificationModelName, falsePositiveCost, targetRecall)
            costReduction = currentCosts - newCosts
            
            print("i = " + str(i) + ", featureId = " + str(featureId) + ", costReduction = " + str(costReduction))
            
            diminishingCostReductionUpperBounds[featureId] = costReduction 
            
            if costReduction >= bestTrueCostReduction:
                bestTrueCostReduction = costReduction
                bestCosts = newCosts
                bestFeatureId = featureId
              
            # lazy greedy  
            if (costReduction >= bestTrueCostReduction) and ((i+1) < costReductionUpperBoundsInOrder.shape[0]) and (costReduction > costReductionUpperBoundsInOrder[i+1]):
                print("!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!! YO WE TOOK SHORT CUT !!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!")
                # assert(False)
                break
        
        # assert(False)
        currentSelectedFeatures.append(bestFeatureId)
        notSelectedFeatureIds_set.remove(bestFeatureId)
        
        allFeatureSetsInOrder.append(numpy.asarray(currentSelectedFeatures, dtype = numpy.int))
        allEstimatedTotalCosts.append(bestCosts)
        currentCosts = bestCosts
    
    # assert(False)
    
    print("found covariate sets = ")
    for i in range(len(allFeatureSetsInOrder)):
        print("covariateIds = " + str(allFeatureSetsInOrder[i]) + " | expected total costs = " + str(allEstimatedTotalCosts[i]))
    
    return allFeatureSetsInOrder, allEstimatedTotalCosts



def getAllFeatureSetsInOrderWithGreedyMethod_lazyWithoutCV(trainData, trainLabels):
    
    allFeatureSetsInOrder = []
    
    p = trainData.shape[1]
    currentSelectedFeatures = []
    notSelectedFeatureIds_set = set(numpy.arange(p))
    
    # add the empty feature set and classifier that does not use any features
    allFeatureSetsInOrder.append(numpy.asarray(currentSelectedFeatures, dtype = numpy.int))
    
    bestScore = float("inf")
    bestFeatureId = None
        
    for featureId in notSelectedFeatureIds_set:
        checkFeatureIds = list(currentSelectedFeatures)
        checkFeatureIds.append(featureId)
        checkFeatureIds = numpy.asarray(checkFeatureIds, dtype = numpy.int)
        assert(len(set(checkFeatureIds)) == len(set(currentSelectedFeatures)) + 1)
        assert(len(set(checkFeatureIds)) == len(checkFeatureIds)) 
        
        _, score = evaluation.getBestGAM_withoutCV(trainData, trainLabels, checkFeatureIds)
        
        if score < bestScore:
            bestScore = score
            bestFeatureId = featureId
        
    
    assert(bestFeatureId is not None)
    currentSelectedFeatures.append(bestFeatureId)
    notSelectedFeatureIds_set.remove(bestFeatureId)
    allFeatureSetsInOrder.append(numpy.asarray(currentSelectedFeatures, dtype = numpy.int))
    currentCosts = bestScore

    diminishingCostReductionUpperBounds = numpy.zeros(p)
    
    for featureId in notSelectedFeatureIds_set:
        checkFeatureIds = list(currentSelectedFeatures)
        checkFeatureIds.append(featureId)
        checkFeatureIds = numpy.asarray(checkFeatureIds, dtype = numpy.int)
        assert(len(set(checkFeatureIds)) == len(set(currentSelectedFeatures)) + 1)
        assert(len(set(checkFeatureIds)) == len(checkFeatureIds)) 
        
        model, score = evaluation.getBestGAM_withoutCV(trainData, trainLabels, checkFeatureIds)
        diminishingCostReductionUpperBounds[featureId] = currentCosts - score
    
    
    while len(notSelectedFeatureIds_set) > 0:
        
        bestTrueCostReduction = float("-inf")
        bestFeatureId = None
        
        notSelectedFeatureIds = numpy.asarray(list(notSelectedFeatureIds_set))
        notSelectedFeatureIds_sortIndex = numpy.argsort(- diminishingCostReductionUpperBounds[notSelectedFeatureIds])
        
        notSelectedFeatureIdsInOrder = notSelectedFeatureIds[notSelectedFeatureIds_sortIndex]
        costReductionUpperBoundsInOrder = diminishingCostReductionUpperBounds[notSelectedFeatureIdsInOrder]
        
        print("**************************")
        print("notSelectedFeatureIds = ", notSelectedFeatureIds)
        print("diminishingCostReductionBounds = ", diminishingCostReductionUpperBounds[notSelectedFeatureIds])
        print("notSelectedFeatureIdsInOrder = ", notSelectedFeatureIdsInOrder)
        print("costReductionUpperBoundsInOrder = ", costReductionUpperBoundsInOrder)
        # assert(False)
        
        for i, featureId in enumerate(notSelectedFeatureIdsInOrder):
            checkFeatureIds = list(currentSelectedFeatures)
            checkFeatureIds.append(featureId)
            checkFeatureIds = numpy.asarray(checkFeatureIds, dtype = numpy.int)
            assert(len(set(checkFeatureIds)) == len(set(currentSelectedFeatures)) + 1)
            assert(len(set(checkFeatureIds)) == len(checkFeatureIds)) 
            
            _, newCosts = evaluation.getBestGAM_withoutCV(trainData, trainLabels, checkFeatureIds)
            costReduction = currentCosts - newCosts
            
            print("i = " + str(i) + ", featureId = " + str(featureId) + ", costReduction = " + str(costReduction))
            
            diminishingCostReductionUpperBounds[featureId] = costReduction 
            
            if costReduction >= bestTrueCostReduction:
                bestTrueCostReduction = costReduction
                bestCosts = newCosts
                bestFeatureId = featureId
              
            # lazy greedy  
            if (costReduction >= bestTrueCostReduction) and ((i+1) < costReductionUpperBoundsInOrder.shape[0]) and (costReduction > costReductionUpperBoundsInOrder[i+1]):
                print("!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!! YO WE TOOK SHORT CUT !!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!")
                # assert(False)
                break
        
        currentSelectedFeatures.append(bestFeatureId)
        notSelectedFeatureIds_set.remove(bestFeatureId)
        allFeatureSetsInOrder.append(numpy.asarray(currentSelectedFeatures, dtype = numpy.int))
        currentCosts = bestCosts
    
    
    print("found covariate sets = ")
    for i in range(len(allFeatureSetsInOrder)):
        print("covariateIds = " + str(allFeatureSetsInOrder[i]) + " | expected total costs = ")
    
    return allFeatureSetsInOrder




def addCurrentFeaturesHelper(currentSelectedFeatures, newFeatures):
    checkFeatureIds = list(currentSelectedFeatures)
    checkFeatureIds.extend(newFeatures)
    checkFeatureIds = numpy.asarray(checkFeatureIds, dtype = numpy.int)
    return checkFeatureIds 


def getBestHalf(trainData, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts, classificationModelName, falsePositiveCost, targetRecall, currentSelectedFeatures, notSelectedFeatureIds):
    assert(len(notSelectedFeatureIds) >= 2)
    halfN = int(len(notSelectedFeatureIds) / 2)
    firstHalfIds = list(notSelectedFeatureIds)[0:halfN]
    secondHalfIds = list(notSelectedFeatureIds)[halfN:len(notSelectedFeatureIds)]
    
    # print("firstHalfIds = ", firstHalfIds)
    # print("secondHalfIds = ", secondHalfIds)
    # assert(False)
    
    checkFeatureIdsCurrentPlusFirstHalf = addCurrentFeaturesHelper(currentSelectedFeatures, firstHalfIds)
    checkFeatureIdsCurrentPlusSecondHalf = addCurrentFeaturesHelper(currentSelectedFeatures, secondHalfIds)
    
    # startTimeTotal = time.time()
    
    allParamsForMultiprocessMap = []
    allParamsForMultiprocessMap.append((trainData, trainLabels, checkFeatureIdsCurrentPlusFirstHalf, misclassificationCosts, definedFeatureCosts, classificationModelName, falsePositiveCost, targetRecall))
    allParamsForMultiprocessMap.append((trainData, trainLabels, checkFeatureIdsCurrentPlusSecondHalf, misclassificationCosts, definedFeatureCosts, classificationModelName, falsePositiveCost, targetRecall))
     
    with Pool(2) as pool:
        allResults = pool.starmap(getOptimalTrainedModel_withMisclassificationCosts, allParamsForMultiprocessMap)
     
    predictedProbsFirstModel, predictionModelFirstHalf, costsFirstHalf = allResults[0]
    predictedProbsSecondModel, predictionModelSecondHalf, costsSecondHalf = allResults[1]
    # print("PRALLEL THREAD, total runtime (in minutes) = " + str((time.time() - startTimeTotal) / 60.0))
    
    # startTimeTotal = time.time()
    # predictedProbsFirstModel, predictionModelFirstHalf, costsFirstHalf = getOptimalTrainedModel_withMisclassificationCosts(trainData, trainLabels, checkFeatureIdsCurrentPlusFirstHalf, misclassificationCosts, definedFeatureCosts, classificationModelName, falsePositiveCost, targetRecall)
    # predictedProbsSecondModel, predictionModelSecondHalf, costsSecondHalf = getOptimalTrainedModel_withMisclassificationCosts(trainData, trainLabels, checkFeatureIdsCurrentPlusSecondHalf, misclassificationCosts, definedFeatureCosts, classificationModelName, falsePositiveCost, targetRecall)
    # print("TWO THREADS, total runtime (in minutes) = " + str((time.time() - startTimeTotal) / 60.0))
    # TWO THREADS, total runtime (in minutes) = 13.964276444911956
    # assert(False)
    
    if costsFirstHalf < costsSecondHalf:
        return predictedProbsFirstModel, predictionModelFirstHalf, firstHalfIds, costsFirstHalf
    else:
        return predictedProbsSecondModel, predictionModelSecondHalf, secondHalfIds, costsSecondHalf
    

def getAllFeatureSetsInOrderWithGreedyMethod_binary(trainData, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts, classificationModelName, falsePositiveCost, targetRecall):
    
    allPredictedProbs = []
    allModels = []
    allFeatureSetsInOrder = []
    allEstimatedTotalCosts = []
    
    # add the empty feature set and classifier that does not use any features
    selectedFeatureIds = numpy.asarray([], dtype = numpy.int)
    # totalCosts = getTotalCostsFromAllData(trainData, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts, selectedFeatureIds, classificationModelName, falsePositiveCost, targetRecall)
    predictedProbsInitialModel, initialModel, totalCosts = getOptimalTrainedModel_withMisclassificationCosts(trainData, trainLabels, selectedFeatureIds, misclassificationCosts, definedFeatureCosts, classificationModelName, falsePositiveCost, targetRecall)
    
    allPredictedProbs.append(predictedProbsInitialModel)
    allModels.append(initialModel)
    allFeatureSetsInOrder.append(selectedFeatureIds)
    allEstimatedTotalCosts.append(totalCosts)
    
    p = trainData.shape[1]
    currentSelectedFeatures = []
    notSelectedFeatureIds = set(numpy.arange(p))
    
    while len(notSelectedFeatureIds) > 0:
        
        if len(notSelectedFeatureIds) == 1:
            
            bestFeatureId = list(notSelectedFeatureIds)[0]
            
            checkFeatureIds = list(currentSelectedFeatures)
            checkFeatureIds.append(bestFeatureId)
            checkFeatureIds = numpy.asarray(checkFeatureIds, dtype = numpy.int)
            assert(len(set(checkFeatureIds)) == len(set(currentSelectedFeatures)) + 1)
            assert(len(set(checkFeatureIds)) == len(checkFeatureIds)) 
            
            #  bestCosts = getTotalCostsFromAllData(trainData, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts, checkFeatureIds, classificationModelName, falsePositiveCost, targetRecall)
            predictedProbsBestModel, bestModel, bestCosts = getOptimalTrainedModel_withMisclassificationCosts(trainData, trainLabels, checkFeatureIds, misclassificationCosts, definedFeatureCosts, classificationModelName, falsePositiveCost, targetRecall)
    
        else:   
            bestHalfIds = list(notSelectedFeatureIds)
            
            while len(bestHalfIds) > 1:
                predictedProbsBestModel, bestModel, bestHalfIds, bestHalfCosts = getBestHalf(trainData, trainLabels, unlabeledData, misclassificationCosts, definedFeatureCosts, classificationModelName, falsePositiveCost, targetRecall, currentSelectedFeatures, bestHalfIds)
            
            assert(len(bestHalfIds) == 1)
            bestFeatureId = bestHalfIds[0] 
            bestCosts = bestHalfCosts
            
        
        currentSelectedFeatures.append(bestFeatureId)
        notSelectedFeatureIds.remove(bestFeatureId)
        
        allFeatureSetsInOrder.append(numpy.asarray(currentSelectedFeatures, dtype = numpy.int))
        allEstimatedTotalCosts.append(bestCosts)
        allModels.append(bestModel)
        allPredictedProbs.append(predictedProbsBestModel)
        
    print("found covariate sets = ")
    for i in range(len(allFeatureSetsInOrder)):
        print("covariateIds = " + str(allFeatureSetsInOrder[i]) + " | expected total costs = " + str(allEstimatedTotalCosts[i]))
    
    return allPredictedProbs, allModels, allFeatureSetsInOrder, allEstimatedTotalCosts



# predictionModel, threshold = getOptimalTrainedModel_withRecallThresholds(trainDataFullCovariates, trainLabels, idsGivenCovariates, targetRecall)
#         allPredictionModels.append(predictionModel)
#         
#         if useTargetRecall:
#             falseNegativeCost = ((1.0 - threshold) / threshold) * falsePositiveCost
#         
#         print("*************")
#         print("idsGivenCovariates = ", idsGivenCovariates)
#         print("falseNegativeCost = ", falseNegativeCost)
#         assert(falsePositiveCost > 0.0)
#         assert(falseNegativeCost > 0.0)
#         
#         # row id = true class id, column id = predicted class id
#         misclassificationCosts = numpy.zeros((2, 2))
#         misclassificationCosts[0, 1] = falsePositiveCost 
#         misclassificationCosts[1, 0] = falseNegativeCost 
#         misclassificationCosts[0, 0] = 0.0
#         misclassificationCosts[1, 1] = 0.0
#         

def getTotalCostsFromAllData_targetRecall(trainData, trainLabels, unlabeledData, falsePositiveCost, definedFeatureCosts, selectedFeatureIds, targetRecall):
    
    predictionModel, threshold = getOptimalTrainedModel_withRecallThresholds(trainData, trainLabels, selectedFeatureIds, targetRecall)
    falseNegativeCost = ((1.0 - threshold) / threshold) * falsePositiveCost
    
    misclassificationCosts = numpy.zeros((2, 2))
    misclassificationCosts[0, 1] = falsePositiveCost 
    misclassificationCosts[1, 0] = falseNegativeCost 
    misclassificationCosts[0, 0] = 0.0
    misclassificationCosts[1, 1] = 0.0

    assert(False) # needs to be updated ?!
    predictionModel, totalCostEstimate = getOptimalTrainedModel_withMisclassificationCosts(trainData, trainLabels, selectedFeatureIds, misclassificationCosts, definedFeatureCosts)
    
    if unlabeledData.shape[0] == 0:
        return totalCostEstimate
    else:
        # this case not yet supported
        assert(False)
        assert(unlabeledData.shape[0] > 0)
        predictedProbs = predictionModel.predict_proba(unlabeledData[:,selectedFeatureIds])
        assert(predictedProbs.shape[0] == unlabeledData.shape[0] and predictedProbs.shape[1] == misclassificationCosts.shape[0])
        return evaluation.bayesRisk(predictedProbs, misclassificationCosts) + numpy.sum(definedFeatureCosts[selectedFeatureIds])



# def getAllFeatureSetsInOrderWithGreedyMethod_targetRecall(trainData, trainLabels, unlabeledData, falsePositiveCost, definedFeatureCosts, targetRecall):
#     
#     allFeatureSetsInOrder = []
#     allEstimatedTotalCosts = []
#     
#     # add the empty feature set and classifier that does not use any features
#     noSelectedFeatureIds = numpy.asarray([], dtype = numpy.int)
#     totalCosts = getTotalCostsFromAllData_targetRecall(trainData, trainLabels, unlabeledData, falsePositiveCost, definedFeatureCosts, noSelectedFeatureIds, targetRecall)
#     allFeatureSetsInOrder.append(noSelectedFeatureIds)
#     allEstimatedTotalCosts.append(totalCosts)
#     
#     p = trainData.shape[1]
#     currentSelectedFeatures = []
#     notSelectedFeatureIds = set(numpy.arange(p))
#     
#     while len(notSelectedFeatureIds) > 0:
#         
#         bestCosts = float("inf")
#         bestFeatureId = None
#     
#         
#         # ORIGINAL PART
#         for featureId in notSelectedFeatureIds:
#             checkFeatureIds = list(currentSelectedFeatures)
#             checkFeatureIds.append(featureId)
#             checkFeatureIds = numpy.asarray(checkFeatureIds, dtype = numpy.int)
#             assert(len(set(checkFeatureIds)) == len(set(currentSelectedFeatures)) + 1)
#             assert(len(set(checkFeatureIds)) == len(checkFeatureIds)) 
#             
#             currentCosts = getTotalCostsFromAllData_targetRecall(trainData, trainLabels, unlabeledData, falsePositiveCost, definedFeatureCosts, checkFeatureIds, targetRecall)
#             if currentCosts < bestCosts:
#                 bestCosts = currentCosts
#                 bestFeatureId = featureId
#         
#         currentSelectedFeatures.append(bestFeatureId)
#         notSelectedFeatureIds.remove(bestFeatureId)
#         
#         allFeatureSetsInOrder.append(numpy.asarray(currentSelectedFeatures, dtype = numpy.int))
#         allEstimatedTotalCosts.append(bestCosts)
# 
#     print("found covariate sets = ")
#     for i in range(len(allFeatureSetsInOrder)):
#         print("covariateIds = " + str(allFeatureSetsInOrder[i]) + " | expected total costs = " + str(allEstimatedTotalCosts[i]))
#     
#     return allFeatureSetsInOrder, allEstimatedTotalCosts




def checkInclusionOrder(allFeatureSetsInOrder):
    # check inclusion order (l1 regularization might not fulfill it)
    for i in range(len(allFeatureSetsInOrder)):
        focusVariableSet = set(allFeatureSetsInOrder[i])
        for j in range(0, i):
            assert(set(allFeatureSetsInOrder[j]) < focusVariableSet)
     
    return   

def filterToEnsureSetInclusionOrder(allFeatureSetsInOrder):
    allFeatureSetsInOrderChecked = []
    for i in range(len(allFeatureSetsInOrder)):
        focusVariableSet = set(allFeatureSetsInOrder[i])
        includesAll = True
        for j in range(0, i):
            if not (set(allFeatureSetsInOrder[j]) < focusVariableSet):
                includesAll = False
                break
        if includesAll:
            allFeatureSetsInOrderChecked.append(numpy.asarray(allFeatureSetsInOrder[i], dtype = numpy.int)) 
     
    allFeatureSetsInOrderCheckedAndRearranged = []
    for i in range(len(allFeatureSetsInOrderChecked)):
        currentFeatureIds = allFeatureSetsInOrderChecked[i]
        if i == 0:
            assert(len(currentFeatureIds) == 0)
            allFeatureSetsInOrderCheckedAndRearranged.append(currentFeatureIds)
        else:
            previousFeatureIds = allFeatureSetsInOrderCheckedAndRearranged[i-1]
            newFeatureIds = list(set(currentFeatureIds) - set(previousFeatureIds))
            updatedFeatureIds = numpy.append(previousFeatureIds, newFeatureIds)
            allFeatureSetsInOrderCheckedAndRearranged.append(updatedFeatureIds)
            
    checkInclusionOrder(allFeatureSetsInOrderCheckedAndRearranged)
    return allFeatureSetsInOrderCheckedAndRearranged
    
    
# use instead "getOptimalTrainedModel_withMisclassificationCosts" 
# def getOptimalTrainedModel(trainDataFullCovariates, trainLabels, selectedFeatureList):
#     if len(selectedFeatureList) == 0:
#         classOneRatio = float(numpy.sum(trainLabels)) / float(trainLabels.shape[0])
#         holdOutDataAccuracyEstimate = numpy.max([classOneRatio, 1.0 - classOneRatio])
#         assert(holdOutDataAccuracyEstimate >= 0.5) 
#         bestLogRegModel = ClassRatioClassifier(trainLabels)
#     else:
#         trainData = trainDataFullCovariates[:, selectedFeatureList]
#         bestLogRegModel, holdOutDataAccuracyEstimate = evaluation.getBestL2RegularizedLogisticRegressionModel(trainData, trainLabels)
#     
#     return bestLogRegModel, holdOutDataAccuracyEstimate


# !NEW! UPDATE
def getOptimalTrainedModel_withTotalCosts(trainDataFullCovariates, trainLabels, selectedFeatureList, misclassificationCosts, definedFeatureCosts, classificationModelName, falsePositiveCost, targetRecall):
    if len(selectedFeatureList) == 0:
        classOneRatio = float(numpy.sum(trainLabels)) / float(trainLabels.shape[0])
        holdOutDataAccuracyEstimate = numpy.max([classOneRatio, 1.0 - classOneRatio])
        assert(holdOutDataAccuracyEstimate >= 0.5) 
        bestModel = ClassRatioClassifier(trainLabels)
        
        predictedProbsBestModel = numpy.ones(trainLabels.shape[0]) * classOneRatio
        
        if misclassificationCosts is not None:
            totalCostEstimate, _ , _, _, _ , _, _, _, _, _, _ = evaluation.getOverallPerformance_fixedCovariateSet(bestModel, trainDataFullCovariates, trainLabels, definedFeatureCosts, misclassificationCosts, selectedFeatureList, targetRecall = 0.95) 
        else:
            _, totalCostEstimate = getAllCosts_fromPredictedTrueProbs(trainLabels, predictedProbsBestModel, selectedFeatureList, definedFeatureCosts, falsePositiveCost, targetRecall = targetRecall)
            # print("totalCostEstimate = ", totalCostEstimate)
            # assert(False)
            
        print("null-mode totalCostEstimate = ", totalCostEstimate)
       
    else:
        trainData = trainDataFullCovariates[:, selectedFeatureList]
        
        bestModel, predictedProbsBestModel = evaluation.getBestModel(trainData, trainLabels, classificationModelName)
        
        if misclassificationCosts is not None:
            predictedLabels = evaluation.bayesClassifier(predictedProbsBestModel, misclassificationCosts)
            avgHoldOutMisclassificationCosts = evaluation.getAverageMisclassificationCosts(trainLabels, predictedLabels, misclassificationCosts)
            
            avgFeatureCosts = numpy.sum(definedFeatureCosts[selectedFeatureList])
            totalCostEstimate = avgHoldOutMisclassificationCosts + avgFeatureCosts 
        else:
            # calculate implicit misclassificationCosts
            _, totalCostEstimate = getAllCosts_fromPredictedTrueProbs(trainLabels, predictedProbsBestModel, selectedFeatureList, definedFeatureCosts, falsePositiveCost, targetRecall = targetRecall)
         
       
        print("non-null-mode totalCostEstimate = ", totalCostEstimate)
        
    
    return predictedProbsBestModel, bestModel, totalCostEstimate



# updated for GAM
def getOptimalTrainedModel_withRecallThresholds(trainDataFullCovariates, trainLabels, selectedFeatureList, targetRecall, classificationModelName):
    assert(targetRecall > 0.0)
    
    if len(selectedFeatureList) == 0:
        classOneRatio = float(numpy.sum(trainLabels)) / float(trainLabels.shape[0])
        holdOutDataAccuracyEstimate = numpy.max([classOneRatio, 1.0 - classOneRatio])
        assert(holdOutDataAccuracyEstimate >= 0.5)
        bestModel = ClassRatioClassifier(trainLabels)
        
        assert(classOneRatio > 0.1)
        threshold = classOneRatio - 0.001 # in order to ensure that always the positive class is selected in case when no covariates are selected. => recall 1 (otherwise we would always have recall 0)
        
        print("null model threshold = ", threshold)
    else:
        trainData = trainDataFullCovariates[:, selectedFeatureList]
        
        if classificationModelName == "logReg":
            bestModel, bestAlphaValue, _ = evaluation.getBestL2RegularizedLogisticRegressionModelNew(trainData, trainLabels)
            threshold = evaluation.getThresholdEstimate(trainData, trainLabels, bestAlphaValue, [targetRecall], modelType = classificationModelName)[0]
        elif classificationModelName == "GAM":
            bestModel = evaluation.getBestGAM(trainData, trainLabels)
            threshold = evaluation.getThresholdEstimate(trainData, trainLabels, bestModel, [targetRecall], modelType = classificationModelName)[0]
        else:
            assert(False)
        
        print("not-null model threshold = ", threshold)
        # assert(False)
        
    return bestModel, threshold





def getAllCosts_fromPredictedTrueProbs(trainLabels, predictedTrueProbs, idsGivenCovariates, definedFeatureCosts, falsePositiveCost, targetRecall = None, falseNegativeCost = None, correctClassificationCost = 0.0):
    assert(falsePositiveCost > 0.0)
    
    if targetRecall is not None:
        assert(falseNegativeCost is None)
        threshold = evaluation.getThresholdFromPredictedProbabilities(trainLabels, predictedTrueProbs, targetRecall)
        falseNegativeCost = ((1.0 - threshold) / threshold) * falsePositiveCost
    
    
    assert(falseNegativeCost > 0.0)
    
    # row id = true class id, column id = predicted class id
    misclassificationCosts = numpy.zeros((2, 2))
    misclassificationCosts[0, 1] = falsePositiveCost 
    misclassificationCosts[1, 0] = falseNegativeCost 
    misclassificationCosts[0, 0] = correctClassificationCost
    misclassificationCosts[1, 1] = correctClassificationCost
    
    predictedLabels = evaluation.bayesClassifier(predictedTrueProbs, misclassificationCosts)
    totalCostEstimate = evaluation.getAverageMisclassificationCosts(trainLabels, predictedLabels, misclassificationCosts) + numpy.sum(definedFeatureCosts[idsGivenCovariates])
     
    return misclassificationCosts, totalCostEstimate


# READING CHECKED
def getAllOperationalCosts_fromPredictedTrueProbs(trainLabels, predictedTrueProbs, idsGivenCovariates, definedFeatureCosts, falsePositiveCost, targetRecall):
    assert(falsePositiveCost > 0.0)
    
    threshold = evaluation.getThresholdFromPredictedProbabilities(trainLabels, predictedTrueProbs, targetRecall)
    
    # row id = true class id, column id = predicted class id
    misclassificationCosts = numpy.zeros((2, 2))
    misclassificationCosts[0, 1] = falsePositiveCost 
    misclassificationCosts[1, 0] = 0.0
    misclassificationCosts[0, 0] = 0.0
    misclassificationCosts[1, 1] = 0.0
    
    predictedLabels = evaluation.thresholdClassifier(predictedTrueProbs, threshold)
    
    # saftey check
    assert(sklearn.metrics.recall_score(trainLabels, predictedLabels) >= targetRecall)
    
    # print("DEBUG INFO:")
    # print("idsGivenCovariates = ", idsGivenCovariates)
    # print("predictedTrueProbs = ", predictedTrueProbs)
    # print("predictedLabels = ", predictedLabels)
    # assert(False)
    
    operationalCostEstimate = evaluation.getAverageMisclassificationCosts(trainLabels, predictedLabels, misclassificationCosts) + numpy.sum(definedFeatureCosts[idsGivenCovariates])
     
    return threshold, operationalCostEstimate


# ensures that p(delta_1 = 1, delta_2 = 1, ..., delta_q = 1 | y=1) >= targetRecall
def getTightThresholdWithRecallGuarrantee(labels, trueProbsAllModels, targetRecall):
    assert(numpy.all(numpy.logical_or(labels == 0, labels == 1)))
    
    # first row is ignored, since this corresponds to no selected variables
    
    probMatrix = numpy.zeros((len(trueProbsAllModels)-1, labels.shape[0]))
    
    for i in range(1, len(trueProbsAllModels), 1):
        probMatrix[i-1] = trueProbsAllModels[i]
    
    probMatrix = probMatrix[:, labels==1]
    
    sortedProbs = numpy.unique(numpy.sort(probMatrix))
    
    
    thresholdId = sortedProbs.shape[0] - 1
    
    
    while(not fullfillsNewRequirement(probMatrix, sortedProbs[thresholdId], targetRecall)):
        thresholdId -= 1
    
    
    return sortedProbs[thresholdId]


# probMatrix = matrix of all probabilities where true label y = 1
def fullfillsNewRequirement(probMatrix, threshold, targetRecall):
            
    labelMatrix = numpy.zeros_like(probMatrix)
    labelMatrix[probMatrix >= threshold] = 1
    
    # labelMatrixExceptLast = labelMatrix[0:labelMatrix.shape[0]-1]
    # allOneRows = numpy.sum(labelMatrixExceptLast, axis = 0) == labelMatrixExceptLast.shape[0]
     
    allOneColumns = numpy.sum(labelMatrix, axis = 0) == labelMatrix.shape[0]
     
    recallLowerBound = numpy.sum(allOneColumns) / labelMatrix.shape[1]
    
    # print("recallLowerBound = ", recallLowerBound)
    # print("labelMatrix = ", labelMatrix.shape)
    # assert(False)
    return recallLowerBound >= targetRecall


    
    
def getThresholdWithRecallGuarrantee(labels, trueProbsAllModels, targetRecall):
    assert(numpy.all(numpy.logical_or(labels == 0, labels == 1)))
    
    # first row is ignored, since this corresponds to no selected variables
    
    probMatrix = numpy.zeros((len(trueProbsAllModels)-1, labels.shape[0]))
    
    for i in range(1, len(trueProbsAllModels), 1):
        probMatrix[i-1] = trueProbsAllModels[i]
    
    probMatrix = probMatrix[:, labels==1]
    
    sortedProbs = numpy.unique(numpy.sort(probMatrix))
    
    thresholdId = sortedProbs.shape[0] - 1
    
    while(not isValidThreshold_checkFromProbMatrix(probMatrix, sortedProbs[thresholdId], targetRecall)):
        thresholdId -= 1
    
    return sortedProbs[thresholdId]


# check once more:
def isValidThreshold_checkFromProbMatrix(probMatrix, threshold, targetRecall):
    
    labelMatrix = numpy.zeros_like(probMatrix)
    labelMatrix[probMatrix >= threshold] = 1
    
    denominator = 1.0
    for i in range(1, labelMatrix.shape[0], 1):
        # print("from classifier " + str(i-1) + " to classifier " + str(i))
        trueIdsPreviousClassifier = labelMatrix[i-1] == 1
        previousClassifierTrueCount = numpy.sum(trueIdsPreviousClassifier) 
        
        if previousClassifierTrueCount > 0:
            previousAndCurrentClassifierTrueCount = numpy.sum(labelMatrix[i, trueIdsPreviousClassifier])
            localStepProb = previousAndCurrentClassifierTrueCount / previousClassifierTrueCount
            # print("localStepProb = ", localStepProb)
            denominator *= localStepProb
    
    
    if denominator == 0.0:
        return False
    
    assert(labelMatrix.shape[1] > 0)
    assert(denominator > 0)
    
    requiredRecall0Classifier = targetRecall / denominator
    recallClassifier0 = numpy.sum(labelMatrix[0]) / labelMatrix.shape[1]
    # print("recall classifier 0 = " + str(recallClassifier0))
    # print("targetRecall = ", targetRecall)
    # print("denominator = ", denominator)
    # print("requiredRecall0Classifier = ", requiredRecall0Classifier)
    
    return recallClassifier0 >= requiredRecall0Classifier

def checkThreshold(labels, trueProbsAllModels, threshold, targetRecall):
    assert(numpy.all(numpy.logical_or(labels == 0, labels == 1)))
    assert(threshold > 0.0 and threshold < 1.0)
     
    # first row is ignored, since this corresponds to no selected variables
    
    probMatrix = numpy.zeros((len(trueProbsAllModels)-1, labels.shape[0]))
    
    for i in range(1, len(trueProbsAllModels), 1):
        probMatrix[i-1] = trueProbsAllModels[i]
    
    probMatrix = probMatrix[:, labels==1]
    
    return isValidThreshold_checkFromProbMatrix(probMatrix, threshold, targetRecall)

        
    

# def getPredictionModelsAndCosts(trainDataFullCovariates, trainLabels, idsGivenCovariates, definedFeatureCosts, falsePositiveCost, targetRecall, useTargetRecall, falseNegativeCost, classificationModelName):
#      
#     if classificationModelName == "Combined":
#         return getPredictionModelsAndCosts_Combined(trainDataFullCovariates, trainLabels, idsGivenCovariates, definedFeatureCosts, falsePositiveCost, targetRecall, useTargetRecall, falseNegativeCost, classificationModelName)
# 
#     predictionModel, threshold = getOptimalTrainedModel_withRecallThresholds(trainDataFullCovariates, trainLabels, idsGivenCovariates, targetRecall, classificationModelName)
#     
#     if useTargetRecall:
#         falseNegativeCost = ((1.0 - threshold) / threshold) * falsePositiveCost
#     
#     print("*************")
#     print("idsGivenCovariates = ", idsGivenCovariates)
#     print("falseNegativeCost = ", falseNegativeCost)
#     assert(falsePositiveCost > 0.0)
#     assert(falseNegativeCost > 0.0)
#     
#     # row id = true class id, column id = predicted class id
#     misclassificationCosts = numpy.zeros((2, 2))
#     misclassificationCosts[0, 1] = falsePositiveCost 
#     misclassificationCosts[1, 0] = falseNegativeCost 
#     misclassificationCosts[0, 0] = 0.0
#     misclassificationCosts[1, 1] = 0.0
#     
#     _, totalCostEstimate = getOptimalTrainedModel_withMisclassificationCosts(trainDataFullCovariates, trainLabels, idsGivenCovariates, misclassificationCosts, definedFeatureCosts, classificationModelName)
#     
#     return predictionModel, misclassificationCosts, totalCostEstimate



# # selects automatically between logReg and GAM
# def getPredictionModelsAndCosts_Combined(trainDataFullCovariates, trainLabels, idsGivenCovariates, definedFeatureCosts, falsePositiveCost, targetRecall, useTargetRecall, falseNegativeCost, classificationModelName):
#     
#     assert(False)
#     
#     predictionModel, threshold = getOptimalTrainedModel_withRecallThresholds(trainDataFullCovariates, trainLabels, idsGivenCovariates, targetRecall, classificationModelName)
#     
#     if useTargetRecall:
#         falseNegativeCost = ((1.0 - threshold) / threshold) * falsePositiveCost
#     
#     print("*************")
#     print("idsGivenCovariates = ", idsGivenCovariates)
#     print("falseNegativeCost = ", falseNegativeCost)
#     assert(falsePositiveCost > 0.0)
#     assert(falseNegativeCost > 0.0)
#     
#     # row id = true class id, column id = predicted class id
#     misclassificationCosts = numpy.zeros((2, 2))
#     misclassificationCosts[0, 1] = falsePositiveCost 
#     misclassificationCosts[1, 0] = falseNegativeCost 
#     misclassificationCosts[0, 0] = 0.0
#     misclassificationCosts[1, 1] = 0.0
#     
#     _, totalCostEstimate = getOptimalTrainedModel_withMisclassificationCosts(trainDataFullCovariates, trainLabels, idsGivenCovariates, misclassificationCosts, definedFeatureCosts, classificationModelName)
#     
#     return predictionModel, misclassificationCosts, totalCostEstimate


# checked
def getAllBestModelsAndTrainingTrueProbs(trainDataFullCovariates, trainLabels, allFeatureListsInOrder, classificationModelName):

    allPredictionModels = []
    allTrainingTrueProbsAllModels = []
    
    for i in range(len(allFeatureListsInOrder)):
        idsGivenCovariates = allFeatureListsInOrder[i]
        
        if len(idsGivenCovariates) == 0:
            classOneRatio = float(numpy.sum(trainLabels)) / float(trainLabels.shape[0])
            holdOutDataAccuracyEstimate = numpy.max([classOneRatio, 1.0 - classOneRatio])
            assert(holdOutDataAccuracyEstimate >= 0.5) 
            bestModel = ClassRatioClassifier(trainLabels)
            trainingTrueProbs = numpy.ones_like(trainLabels) * classOneRatio
            
            # print("trainingTrueProbs = ")
            # print(trainingTrueProbs)
            # assert(False)
        else:
            trainData = trainDataFullCovariates[:, idsGivenCovariates]
            bestModel, trainingTrueProbs = evaluation.getBestModel(trainData, trainLabels, classificationModelName)
     
        allPredictionModels.append(bestModel)
        allTrainingTrueProbsAllModels.append(trainingTrueProbs)
        
    
    return allPredictionModels, allTrainingTrueProbsAllModels


    
# updated for GAM
# def getAllPredictionModelsAndCosts(trainDataFullCovariates, trainLabels, allFeatureListsInOrder, definedFeatureCosts, falsePositiveCost, targetRecall, useTargetRecall, falseNegativeCost, classificationModelName):
#     
#     allPredictionModels = []
#     numberOfFeatureSets = len(allFeatureListsInOrder)
#     
#     allMisclassificationCosts = []
#     allTotalCosts = []
#     
#     for i in range(numberOfFeatureSets):
#         idsGivenCovariates = allFeatureListsInOrder[i]
#         predictionModel, misclassificationCosts, totalCostEstimate = getPredictionModelsAndCosts(trainDataFullCovariates, trainLabels, idsGivenCovariates, definedFeatureCosts, falsePositiveCost, targetRecall, useTargetRecall, falseNegativeCost, classificationModelName)
#         allPredictionModels.append(predictionModel)
#         allMisclassificationCosts.append(misclassificationCosts)
#         allTotalCosts.append(totalCostEstimate)
#         
#     print("found covariate sets = ")
#     for i in range(len(allFeatureListsInOrder)):
#         print("covariateIds = " + str(allFeatureListsInOrder[i]) + " | expected total costs = " + str(allTotalCosts[i]))
#      
#     # assert(False)
#     
#     return allPredictionModels, allMisclassificationCosts, allTotalCosts


# def getTotalCostEstimateAndTrainedModel(trainDataFullCovariates, trainLabels, misclassificationCosts, definedFeatureCosts, selectedFeatureList):
#     assert(numpy.all(numpy.logical_or(trainLabels == 0, trainLabels == 1)))
#     assert(misclassificationCosts[0,1] == misclassificationCosts[1,0])
#     
#     bestLogRegModel, holdOutDataAccuracyEstimate = getOptimalTrainedModel(trainDataFullCovariates, trainLabels, selectedFeatureList)
#     
#     misclassificationCostSymmetric = misclassificationCosts[0,1]
#     avgTotalFeatureCosts = numpy.sum(definedFeatureCosts[selectedFeatureList])
#     totalCostEstimate = evaluation.getTotalCostsSimple(holdOutDataAccuracyEstimate, avgTotalFeatureCosts, misclassificationCostSymmetric)
#     
#     # print("definedFeatureCosts = ", definedFeatureCosts)
#     # print("avgTotalFeatureCosts = ", avgTotalFeatureCosts)
#         
#     return totalCostEstimate, bestLogRegModel


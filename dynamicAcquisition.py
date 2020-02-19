import numpy
import prepareFeatureSets
import oneDimIntegral
import sklearn.linear_model
import sklearn.gaussian_process
import experimentHelper
import evaluation
import pygam


def getConditionalMeanAndVariance(allData, idsGivenCovariates, idsQueryCovariates):
    p = allData.shape[1]
    idsMarginalizeOutCovariates = numpy.arange(p)
    idsMarginalizeOutCovariates = numpy.delete(idsMarginalizeOutCovariates, numpy.hstack((idsQueryCovariates, idsGivenCovariates)))
    assert(len(set(idsMarginalizeOutCovariates) | set(idsQueryCovariates) | set(idsGivenCovariates)) == p)
           
    # print("idsMarginalizeOutCovariates = ", idsMarginalizeOutCovariates)
    # print("idsGivenCovariates = ", idsGivenCovariates)
    # print("idsQueryCovariates = ", idsQueryCovariates)
        
    orderedIds = numpy.hstack((idsQueryCovariates, idsMarginalizeOutCovariates, idsGivenCovariates))
    assert(orderedIds.shape[0] == p)
    
    allDataOrdered = allData[:, orderedIds]
    covMatrix = numpy.cov(allDataOrdered.transpose(), bias=True)
    
    upperLeftBlockSize = idsQueryCovariates.shape[0] + idsMarginalizeOutCovariates.shape[0]
    
    upperRightBlock = covMatrix[0:upperLeftBlockSize, upperLeftBlockSize:p]
    upperLeftBlock = covMatrix[0:upperLeftBlockSize, 0:upperLeftBlockSize]
    lowerRightBlock = covMatrix[upperLeftBlockSize:p, upperLeftBlockSize:p]
    
    # relevant matrices for the characterization of the normal distribution of rest variables given "idsGivenCovariates" variables
    sigma12TimesSigma22Inv = numpy.matmul(upperRightBlock, numpy.linalg.inv(lowerRightBlock))
    newCovMatrix = upperLeftBlock - numpy.matmul(sigma12TimesSigma22Inv, upperRightBlock.transpose())
    
    # experimentHelper.showMatrix(sigma12TimesSigma22Inv)
    # experimentHelper.showMatrix(newCovMatrix)
    
    # get the variables from query only
    q = idsQueryCovariates.shape[0]
    sigma12TimesSigma22InvQueryPart = sigma12TimesSigma22Inv[0:q, :]
    newCovMatrixQueryPart = newCovMatrix[0:q, 0:q]
    
    assert(sigma12TimesSigma22Inv.shape[1] == len(idsGivenCovariates))
    return sigma12TimesSigma22InvQueryPart, newCovMatrixQueryPart


# use this for the ordinary linear regression model as density model 
def prepareSamplerAndClassifierForTest(allData, allFeatureListsInOrder):
    
    allSamplerInfos = []
    numberOfFeatureSets = len(allFeatureListsInOrder)
        
    for i in range(numberOfFeatureSets):
        idsGivenCovariates = allFeatureListsInOrder[i]
        
        # print("******************** i = " + str(i) + " ***********************")
                 
        samplerInfo = []
        for j in range(i + 1, numberOfFeatureSets):
            idsNextCovariates = allFeatureListsInOrder[j]
            # assert((trainDataFullCovariates[:, idsNextCovariates]).shape[1] > 0)
            
            # idsQueryCovariates = list(set(idsNextCovariates) - set(idsGivenCovariates))
            # idsQueryCovariates.sort()  # just to make sure that the order is always the same (though might not be necessary)
            # idsQueryCovariates = numpy.asarray(idsQueryCovariates, dtype=numpy.int)
            
            idsQueryCovariates = idsNextCovariates[idsGivenCovariates.shape[0] : idsNextCovariates.shape[0]]
            # print("idsQueryCovariates = ", idsQueryCovariates)
            
            sigma12TimesSigma22InvQueryPart, newCovMatrixQueryPart = getConditionalMeanAndVariance(allData, idsGivenCovariates, idsQueryCovariates)
            assert(sigma12TimesSigma22InvQueryPart.shape[0] == idsQueryCovariates.shape[0])
            assert(sigma12TimesSigma22InvQueryPart.shape[1] == idsGivenCovariates.shape[0])
            samplerInfo.append((idsQueryCovariates, sigma12TimesSigma22InvQueryPart, newCovMatrixQueryPart))

        # print("allData.mean = ", numpy.mean(allData, axis = 0))
        # assert(False)
        
        allSamplerInfos.append(samplerInfo)
        
    return allSamplerInfos



def getRegressionModel(densityTrainDataCovariates, densityTrainDataResponse, name, baseVariance):
    
    if densityTrainDataCovariates.shape[1] == 0:
        return experimentHelper.BaseRegressionModel(0.0, baseVariance)
    elif numpy.all(densityTrainDataResponse == densityTrainDataResponse[0]):
        baseMean = densityTrainDataResponse[0]
        return experimentHelper.BaseRegressionModel(baseMean, 0.0)

    
    if name == "OrdinaryRegression":
        # print("use my original implementation")
        # assert(False)
        model = sklearn.linear_model.LinearRegression()
    elif name == "BR":
        model = sklearn.linear_model.BayesianRidge(verbose=True)
    elif name == "GP":
        covFuncRBF = sklearn.gaussian_process.kernels.RBF(length_scale = 1.0)
        covFuncConst = sklearn.gaussian_process.kernels.ConstantKernel(constant_value=1.0)
        covFuncFinal = sklearn.gaussian_process.kernels.Product(covFuncConst, covFuncRBF)
        model = sklearn.gaussian_process.GaussianProcessRegressor(kernel=covFuncFinal, alpha=1.0) # , optimizer=None, normalize_y=False)
    else:
        assert(False)
        
    model.fit(densityTrainDataCovariates, densityTrainDataResponse)
    
    return model



# updated for GAM
# use this for the other regression models as density model
def prepareSamplerAndClassifierForTestWithRegression(allPredictionModels, allData, allFeatureListsInOrder, densityRegressionModelName):
    
    numberOfFeatureSets = len(allFeatureListsInOrder)

    allSamplerInfos = []
    
    for i in range(numberOfFeatureSets):
        idsGivenCovariates = allFeatureListsInOrder[i]
                  
        samplerInfo = []
        for j in range(i + 1, numberOfFeatureSets):
            idsNextCovariates = allFeatureListsInOrder[j]
            # assert((trainDataFullCovariates[:, idsNextCovariates]).shape[1] > 0)
            
            # idsQueryCovariates = list(set(idsNextCovariates) - set(idsGivenCovariates))
            # idsQueryCovariates.sort()  # just to make sure that the order is always the same (though might not be necessary)
            # idsQueryCovariates = numpy.asarray(idsQueryCovariates, dtype=numpy.int)
            
            idsQueryCovariates = idsNextCovariates[idsGivenCovariates.shape[0] : idsNextCovariates.shape[0]]
            # print("idsQueryCovariates = ", idsQueryCovariates)
            
            nextPredictionModel = allPredictionModels[j] 
            densityTrainDataCovariates = allData[:, idsGivenCovariates]
            
            # print(isinstance(nextPredictionModel, pygam.pygam.LogisticGAM))
            # print(isinstance(nextPredictionModel, sklearn.linear_model.logistic.LogisticRegression))
    
            if isinstance(nextPredictionModel, sklearn.linear_model.logistic.LogisticRegression):
                beta = nextPredictionModel.coef_[0]
                assert(beta.shape[0] == idsNextCovariates.shape[0])
                queryBetaPart = beta[idsGivenCovariates.shape[0] : idsNextCovariates.shape[0]]
                
                densityTrainDataResponse = allData[:, idsQueryCovariates] @ queryBetaPart
                
                covMatrix = numpy.cov((allData[:, idsQueryCovariates]).transpose())
                
            elif isinstance(nextPredictionModel, pygam.pygam.LogisticGAM):
                assert(len(nextPredictionModel.terms._terms) == idsNextCovariates.shape[0] + 1)
                NR_SPLINES = 20
                assert(numpy.all(numpy.asarray(nextPredictionModel.n_splines) == NR_SPLINES))
                assert(nextPredictionModel.coef_.shape[0] == NR_SPLINES * idsNextCovariates.shape[0] + 1)
                startId = NR_SPLINES * idsGivenCovariates.shape[0]
                endId = NR_SPLINES * idsNextCovariates.shape[0]
                queryBetaPart = nextPredictionModel.coef_[startId:endId]
                
                splineBasis = nextPredictionModel.terms.build_columns(allData[:, idsNextCovariates])
                splineBasis = splineBasis.todense()
                queryData_splineBasis = splineBasis[:, startId:endId]
                
                densityTrainDataResponse = numpy.matmul(queryData_splineBasis, queryBetaPart)
                assert(densityTrainDataResponse.shape[0] == 1 and densityTrainDataResponse.shape[1] == allData.shape[0])
                densityTrainDataResponse = numpy.asarray(densityTrainDataResponse)[0]
                
                covMatrix = numpy.cov(queryData_splineBasis.transpose())
                
                # print("queryBetaPart.shape = ", queryBetaPart.shape)
                # print("queryData_splineBasis.shape = ", queryData_splineBasis.shape)
                # print("densityTrainDataResponse.shape = ", densityTrainDataResponse.shape)
                # print("covMatrix.shape = ", covMatrix.shape)
                # assert(False)
            else:
                assert(False)
                
            
            if len(covMatrix.shape) == 0:
                # covMatrix is a scalar
                baseVariance = covMatrix * (queryBetaPart.transpose() @ queryBetaPart)
            else:
                assert(covMatrix.shape[0] == queryBetaPart.shape[0])
                baseVariance = queryBetaPart.transpose() @ covMatrix @ queryBetaPart
              
            assert(not numpy.any(numpy.isnan(densityTrainDataCovariates)))  
            assert(not numpy.any(numpy.isnan(densityTrainDataResponse)))
            assert(not numpy.isnan(baseVariance))
            
            densityRegressionModel = getRegressionModel(densityTrainDataCovariates, densityTrainDataResponse, densityRegressionModelName, baseVariance)

            allSameValue = numpy.all(densityTrainDataResponse == densityTrainDataResponse[0])
            
            if allSameValue:
                print("WARNING: response all same value !")
                print("j = " + str(j) + " out of " + str(numberOfFeatureSets))
                print("!!! THERE MIGHT BE CHANCE TO IMPROVEMENT !!!")
                # assert(j == numberOfFeatureSets - 1)

            
            # test regression model
#             for i in range(densityTrainDataCovariates.shape[0]):
#                 observedCovariatesForClassifier = numpy.atleast_2d( numpy.random.randn(densityTrainDataCovariates.shape[1]) )
#                 allMeans, allSTDs = densityRegressionModel.predict(observedCovariatesForClassifier, return_std=True)
#                 assert(allMeans.shape[0] == 1 and allSTDs.shape[0] == 1)
#                 mean = allMeans[0]
#                 variance = numpy.square(allSTDs[0])
#                     
#                 if numpy.isnan(mean) or numpy.isnan(variance):
#                     print("! ERROR HERE ! ")
#                     print("allMeans = ", allMeans)
#                     print("allSTDs = ", allSTDs)
#                     print("observedCovariatesForClassifier = ", observedCovariatesForClassifier)
#                     print("densityTrainDataCovariates = ", densityTrainDataCovariates)
#                     print("densityTrainDataResponse = ", densityTrainDataResponse)
#                     assert(False)
                    
            
            samplerInfo.append((idsQueryCovariates, densityRegressionModel))
        
        allSamplerInfos.append(samplerInfo)
        
    
    # assert(False)
    
    return allSamplerInfos



# checked
# def getExpectedBayesRiskMCEstimateSlow(misclassificationCosts, observedCovariates, meanQueryPart, covQueryPart, predictionModel, NUMBER_OF_SAMPLES):
#     assert(covQueryPart.shape[0] == covQueryPart.shape[1])
#     assert(meanQueryPart.shape[0] == covQueryPart.shape[0])
#     
#     allSamples = numpy.random.multivariate_normal(mean=meanQueryPart, cov=covQueryPart, size=NUMBER_OF_SAMPLES)
#     assert(allSamples.shape[0] == NUMBER_OF_SAMPLES and allSamples.shape[1] == meanQueryPart.shape[0])
#     
#     totalBayesRisk = 0.0
#     for i in range(NUMBER_OF_SAMPLES):
#         queryCovariatesMCSample = allSamples[i]
#         assert(queryCovariatesMCSample.shape[0] == meanQueryPart.shape[0])
#         newCovariateVec = numpy.hstack((observedCovariates, queryCovariatesMCSample))
#         newCovariateVecForClassifier = numpy.reshape(newCovariateVec, (1, -1))  # classifier expects a 2D array
#         predictedProbs = predictionModel.predict_proba(newCovariateVecForClassifier)
#         assert(predictedProbs.shape[0] == 1 and predictedProbs.shape[1] == misclassificationCosts.shape[0])
#         totalBayesRisk += prepareFeatureSets.bayesRisk(predictedProbs, misclassificationCosts)
#     
#     return totalBayesRisk / float(NUMBER_OF_SAMPLES)


def getFixedContribution(observedCovariates, predictionModel):
    
    if isinstance(predictionModel, sklearn.linear_model.logistic.LogisticRegression):
        beta = predictionModel.coef_[0]
        intercept = predictionModel.intercept_[0]
        
        observedBetaPart = beta[0:observedCovariates.shape[0]]
        fixedContribution = numpy.dot(observedBetaPart, observedCovariates) + intercept
        
    elif isinstance(predictionModel, pygam.pygam.LogisticGAM):
        fullNumberOfCovariates = len(predictionModel.terms._terms) - 1
        assert(fullNumberOfCovariates > observedCovariates.shape[0])
        NR_SPLINES = 20
        assert(numpy.all(numpy.asarray(predictionModel.n_splines) == NR_SPLINES))
        assert(predictionModel.coef_.shape[0] == NR_SPLINES * fullNumberOfCovariates + 1)
        intercept = predictionModel.coef_[NR_SPLINES * fullNumberOfCovariates]
        
        paddedCovariateVector = numpy.zeros((1,fullNumberOfCovariates))
        paddedCovariateVector[0, 0:observedCovariates.shape[0]] = observedCovariates
        
        # print("paddedCovariateVector = ")
        # print(paddedCovariateVector)
        
        endId = NR_SPLINES * observedCovariates.shape[0]
        observedBetaPart = predictionModel.coef_[0:endId]
        
        splineBasis = predictionModel.terms.build_columns(paddedCovariateVector)
        splineBasis = splineBasis.todense()
        oberservedCovariates_splineBasis = splineBasis[:, 0:endId]
        
        # print("intercept = ", intercept)
        # print("observedBetaPart = ", observedBetaPart.shape)
        # print("oberservedCovariates_splineBasis = ", oberservedCovariates_splineBasis.shape)
        
        fixedContribution = numpy.matmul(oberservedCovariates_splineBasis, observedBetaPart) + intercept
        
        assert(fixedContribution.shape[0] == 1 and fixedContribution.shape[1] == 1)
        fixedContribution = numpy.asarray(fixedContribution)[0,0]
        
        # print("fixedContribution = ", fixedContribution)
        # assert(False)
        
    else:
        assert(False)
    
    return fixedContribution


def getExpectedFutureBayesRiskEstimate(misclassificationCosts, observedCovariates, mean, variance, predictionModel):
    
    fixedContribution = getFixedContribution(observedCovariates, predictionModel)
    
    zStar = numpy.log(misclassificationCosts[0,1] / misclassificationCosts[1,0]) - fixedContribution
    sigma = numpy.sqrt(variance)
    
    class1ExpectationEst = misclassificationCosts[1,1] * oneDimIntegral.getIntegralEstimate(mean + fixedContribution, sigma, zStar + fixedContribution, float("inf"))
    class1ExpectationEst += misclassificationCosts[1,0] * oneDimIntegral.getIntegralEstimate(mean + fixedContribution, sigma, float("-inf"), zStar + fixedContribution)
    
    class2ExpectationEst = misclassificationCosts[0,0] * oneDimIntegral.getNormalCDFpart(mean, sigma, float("-inf"), zStar)
    class2ExpectationEst -= misclassificationCosts[0,0] * oneDimIntegral.getIntegralEstimate(mean + fixedContribution, sigma, float("-inf"), zStar + fixedContribution)
    
    class2ExpectationEst += misclassificationCosts[0,1] * oneDimIntegral.getNormalCDFpart(mean, sigma, zStar, float("inf"))
    class2ExpectationEst -= misclassificationCosts[0,1] * oneDimIntegral.getIntegralEstimate(mean + fixedContribution, sigma, zStar + fixedContribution, float("inf"))
    
    return class1ExpectationEst + class2ExpectationEst


# checked for applicability to target recall
def getExpectedFutureFalsePositiveCostEstimate(threshold, falsePositiveCost, observedCovariates, mean, variance, predictionModel):
    assert(threshold > 0.0 and threshold < 1.0)
    
    fixedContribution = getFixedContribution(observedCovariates, predictionModel)
    
    zStar = - numpy.log((1.0/threshold) - 1) - fixedContribution
    sigma = numpy.sqrt(variance)
    
    class2ExpectationEst = falsePositiveCost * oneDimIntegral.getNormalCDFpart(mean, sigma, zStar, float("inf"))
    class2ExpectationEst -= falsePositiveCost * oneDimIntegral.getIntegralEstimate(mean + fixedContribution, sigma, zStar + fixedContribution, float("inf"))
    
    return class2ExpectationEst


# checked for applicability to target recall
# checked
def getExpectedBayesRiskMCEstimateFast(misclassificationCosts, observedCovariates, mean, variance, predictionModel, NUMBER_OF_SAMPLES):
    
    beta = predictionModel.coef_[0]
    intercept = predictionModel.intercept_[0]
    observedBetaPart = beta[0:observedCovariates.shape[0]]
    # queryBetaPart = beta[observedCovariates.shape[0]:beta.shape[0]]
    fixedContribution = numpy.dot(observedBetaPart, observedCovariates) + intercept
     
    # mean = numpy.dot(queryBetaPart, meanQueryPart)
    # variance = queryBetaPart @ covQueryPart @ queryBetaPart.transpose()
    # print("mean = ", mean)
    # print("variance = ", variance)
    assert(len(mean.shape) == 0)
    assert(len(variance.shape) == 0)
    allSamples = numpy.random.normal(loc=mean, scale=numpy.sqrt(variance), size=NUMBER_OF_SAMPLES)
    assert(len(allSamples.shape) == 1 and allSamples.shape[0] == NUMBER_OF_SAMPLES)
     
    totalBayesRisk = 0.0
    for i in range(NUMBER_OF_SAMPLES):
        z = allSamples[i]
        # queryCovariatesMCSample = allSamples[i]
        # assert(queryCovariatesMCSample.shape[0] == meanQueryPart.shape[0])
        # newCovariateVec = numpy.hstack((observedCovariates, queryCovariatesMCSample))
        # newCovariateVecForClassifier = numpy.reshape(newCovariateVec, (1,-1)) # classifier expects a 2D array
        # mcContribution = numpy.dot(queryBetaPart, queryCovariatesMCSample)
        class1Prob = 1.0 / (1.0 + numpy.exp(-(z + fixedContribution)))
        predictedProbs = numpy.asarray([1.0 - class1Prob, class1Prob])
        predictedProbs = numpy.reshape(predictedProbs, (1, -1))
        # print("myProb = ", predictedProbs) 
         
        # predictedTrueProbs = predictionModel.predict_proba(newCovariateVecForClassifier)
        # print("true probs = ", predictedTrueProbs)
        # assert(False)
        # predictedProbs = predictionModel.predict_proba(newCovariateVecForClassifier)
         
        assert(predictedProbs.shape[0] == 1 and predictedProbs.shape[1] == misclassificationCosts.shape[0])
        totalBayesRisk += evaluation.bayesRisk(predictedProbs, misclassificationCosts)
     
    return totalBayesRisk / float(NUMBER_OF_SAMPLES)



# checked
# def runDynamicAcquisition(definedFeatureCosts, misclassificationCosts, allFeatureSetsInOrder, allSamplerInfos, allPredictionModels, testSample, onlyLookingOneStepAhead, NUMBER_OF_SAMPLES):
#     assert(len(testSample.shape) == 1)
#     assert(testSample.shape[0] == definedFeatureCosts.shape[0])
#     assert(len(allFeatureSetsInOrder) == len(allSamplerInfos))
#     assert(len(allFeatureSetsInOrder) == len(allPredictionModels))
#     
#     # print("Analyze new test sample")
#     
#     numberOfFeatureSets = len(allFeatureSetsInOrder)
#     
#     for i in range(numberOfFeatureSets):
#         idsGivenCovariates = allFeatureSetsInOrder[i]
#         observedCovariates = testSample[idsGivenCovariates]
#         currentFeatureCosts = numpy.sum(definedFeatureCosts[idsGivenCovariates])
#         
#         # print("idsGivenCovariates = ", idsGivenCovariates)
#         
#         # handle the special case that all ids are False, i.e. no covariates are selected
#         # if observedCovariates.shape[0] == 0:
#         #    assert(numpy.all(numpy.bitwise_not(idsGivenCovariates)))
#         #    idsGivenCovariates = numpy.zeros((0), dtype=numpy.int)
#         
#         currentPredictionModel = allPredictionModels[i]
#         
#         observedCovariatesForClassifier = numpy.reshape(observedCovariates, (1, -1))  # classifier expects a 2D array
#         predictedProbs = currentPredictionModel.predict_proba(observedCovariatesForClassifier)
#         assert(predictedProbs.shape[0] == 1 and predictedProbs.shape[1] == misclassificationCosts.shape[0])
#         currentBayesRisk = prepareFeatureSets.bayesRisk(predictedProbs, misclassificationCosts)
#         
#         # print("currentBayesRisk = ", currentBayesRisk)
#         
#         continueAskingForFeatures = False
#         
#         for j, sampleInfoTriple in enumerate(allSamplerInfos[i]):
#             
#             nextPredictionModel = allPredictionModels[i + j + 1]
#             
#             if len(sampleInfoTriple) == 3:
#                 queryCovariateIds, sigma12TimesSigma22InvQueryPart, newCovMatrixQueryPart = sampleInfoTriple
#                 assert(sigma12TimesSigma22InvQueryPart.shape[1] == observedCovariates.shape[0] and sigma12TimesSigma22InvQueryPart.shape[0] == queryCovariateIds.shape[0])
#                 meanQueryPart = numpy.matmul(sigma12TimesSigma22InvQueryPart, observedCovariates)
#                 
#                 assert(len(meanQueryPart.shape) == 1)
#                 assert(meanQueryPart.shape[0] > 0)
#                 assert(meanQueryPart.shape[0] == queryCovariateIds.shape[0])
#                 
#                 assert(sigma12TimesSigma22InvQueryPart.shape[0] > 0)
#                 if sigma12TimesSigma22InvQueryPart.shape[1] == 0:
#                     assert(idsGivenCovariates.shape[0] == 0)
#                     # this case means no given covariates
#                     assert(numpy.all(meanQueryPart == 0))  # we assume that each covariate has zero mean
#                 assert(newCovMatrixQueryPart.shape[0] > 0 and newCovMatrixQueryPart.shape[1] > 0)
#                   
#                 
#                 # print("allFeatureSetsInOrder[i + j + 1] = ", allFeatureSetsInOrder[i + j + 1])
#                 # print("numpy.append(idsGivenCovariates, queryCovariateIds) = ", numpy.append(idsGivenCovariates, queryCovariateIds))
#                 assert(numpy.all(numpy.equal(allFeatureSetsInOrder[i + j + 1], numpy.append(idsGivenCovariates, queryCovariateIds))))
#                 assert(nextPredictionModel.coef_.shape[1] == idsGivenCovariates.shape[0] + queryCovariateIds.shape[0])
#                   
#                 beta = nextPredictionModel.coef_[0]
#                 queryBetaPart = beta[observedCovariates.shape[0]:beta.shape[0]]
#                 
#                 mean = numpy.dot(queryBetaPart, meanQueryPart)
#                 variance = queryBetaPart @ newCovMatrixQueryPart @ queryBetaPart.transpose()
#                 
#             else:
#                 assert(len(sampleInfoTriple) == 2)
#                 queryCovariateIds, densityRegModel = sampleInfoTriple
#                 allMeans, allSTDs = densityRegModel.predict(observedCovariatesForClassifier, return_std=True)
#                 assert(allMeans.shape[0] == 1 and allSTDs.shape[0] == 1)
#                 mean = allMeans[0]
#                 variance = numpy.square(allSTDs[0])
#             
#             if NUMBER_OF_SAMPLES is not None:
#                 expectedFutureBayesRisk = getExpectedBayesRiskMCEstimateFast(misclassificationCosts, observedCovariates, mean, variance, nextPredictionModel, NUMBER_OF_SAMPLES)
#             else:
#                 expectedFutureBayesRisk = getExpectedBayesRiskEstimateProposed(misclassificationCosts, observedCovariates, mean, variance, nextPredictionModel)
#             
#             additionalFeatureCosts = numpy.sum(definedFeatureCosts[queryCovariateIds])
#            
#             # print("queryCovariateIds = " + str(queryCovariateIds) + ", expectedFutureBayesRisk = " + str(expectedFutureBayesRisk))
#             totalExpectedFutureCosts = expectedFutureBayesRisk + additionalFeatureCosts  
#             
#             
#             if totalExpectedFutureCosts < currentBayesRisk:
#                 continueAskingForFeatures = True
#                 break
#             
#             if onlyLookingOneStepAhead:
#                 break 
#         
#         # assert(False)
#         
#         if not continueAskingForFeatures:
#             # print("idsGivenCovariates = ", idsGivenCovariates)
#             # assert(False)
#             predictedProbs = currentPredictionModel.predict_proba(observedCovariatesForClassifier)
#             predictedLabel = currentPredictionModel.predict(observedCovariatesForClassifier)
#             
#             assert(len(predictedLabel.shape) == 1 and predictedLabel.shape[0] == 1)
#             return idsGivenCovariates, currentFeatureCosts, predictedLabel[0], predictedProbs[0,1]
#     
#     assert(False)
#     return None



# formerly "runDynamicAcquisition(..)
def runDynamicAcquisition_bayesRisk(sampleId, definedFeatureCosts, allFeatureSetsInOrder, allSamplerInfos, allPredictionModels, allMisclassificationCosts, testSample, onlyLookingOneStepAhead, NUMBER_OF_SAMPLES):
    assert(len(testSample.shape) == 1)
    assert(testSample.shape[0] == definedFeatureCosts.shape[0])
    assert(len(allFeatureSetsInOrder) == len(allSamplerInfos))
    assert(len(allFeatureSetsInOrder) == len(allPredictionModels))
    
    print("Process test sample ", sampleId)
    
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
        currentMisclassificationCosts = allMisclassificationCosts[i]
        
        observedCovariatesForClassifier = numpy.reshape(observedCovariates, (1, -1))  # classifier expects a 2D array
        predictedProbs = currentPredictionModel.predict_proba(observedCovariatesForClassifier)
        predictedProbs = evaluation.ensure2D(predictedProbs)
        
        # predictedProbs = numpy.atleast_2d(predictedProbs)
        
        # if not (predictedProbs.shape[0] == 1 and predictedProbs.shape[1] == currentMisclassificationCosts.shape[0]):
#         if len(predictedProbs.shape) < 2:
#             print("! observedCovariatesForClassifier.shape = ", observedCovariatesForClassifier.shape)
#             print("! currentMisclassificationCosts = ", currentMisclassificationCosts)
#             print("! predictedProbs = ", predictedProbs)
#             print("! predictedProbs.shape = ", predictedProbs.shape)
#             print("! predictedProbs = ", predictedProbs)
#             
#             ! observedCovariatesForClassifier.shape =  (1, 3)
# ! currentMisclassificationCosts =  [[ 0.         10.        ]
#  [ 8.28237866  0.        ]]
# ! predictedProbs =  [0.01997131]
# ! predictedProbs.shape =  (1,)
# ! predictedProbs =  [0.01997131]
# 
#         else:
#             print("OK")
#             print("OK observedCovariatesForClassifier.shape = ", observedCovariatesForClassifier.shape)
#             print("OK currentMisclassificationCosts = ", currentMisclassificationCosts)
#             print("OK predictedProbs = ", predictedProbs)
           
            
        assert(predictedProbs.shape[0] == 1 and predictedProbs.shape[1] == currentMisclassificationCosts.shape[0])
        
        currentBayesRisk = evaluation.bayesRisk(predictedProbs, currentMisclassificationCosts)
        
        # print("currentBayesRisk = ", currentBayesRisk)
        
        continueAskingForFeatures = False
        
        for j, sampleInfoTriple in enumerate(allSamplerInfos[i]):
            
            nextPredictionModel = allPredictionModels[i + j + 1]
            nextMisclassifcationCosts = allMisclassificationCosts[i + j + 1]
            
            if len(sampleInfoTriple) == 3:
                
                assert(False) # because not valid anymore for GAM               
                queryCovariateIds, sigma12TimesSigma22InvQueryPart, newCovMatrixQueryPart = sampleInfoTriple
                assert(sigma12TimesSigma22InvQueryPart.shape[1] == observedCovariates.shape[0] and sigma12TimesSigma22InvQueryPart.shape[0] == queryCovariateIds.shape[0])
                meanQueryPart = numpy.matmul(sigma12TimesSigma22InvQueryPart, observedCovariates)
                
                assert(len(meanQueryPart.shape) == 1)
                assert(meanQueryPart.shape[0] > 0)
                assert(meanQueryPart.shape[0] == queryCovariateIds.shape[0])
                
                assert(sigma12TimesSigma22InvQueryPart.shape[0] > 0)
                if sigma12TimesSigma22InvQueryPart.shape[1] == 0:
                    assert(idsGivenCovariates.shape[0] == 0)
                    # this case means no given covariates
                    assert(numpy.all(meanQueryPart == 0))  # we assume that each covariate has zero mean
                assert(newCovMatrixQueryPart.shape[0] > 0 and newCovMatrixQueryPart.shape[1] > 0)
                  
                
                # print("allFeatureSetsInOrder[i + j + 1] = ", allFeatureSetsInOrder[i + j + 1])
                # print("numpy.append(idsGivenCovariates, queryCovariateIds) = ", numpy.append(idsGivenCovariates, queryCovariateIds))
                assert(numpy.all(numpy.equal(allFeatureSetsInOrder[i + j + 1], numpy.append(idsGivenCovariates, queryCovariateIds))))
                assert(nextPredictionModel.coef_.shape[1] == idsGivenCovariates.shape[0] + queryCovariateIds.shape[0])
                  
                beta = nextPredictionModel.coef_[0]
                queryBetaPart = beta[observedCovariates.shape[0]:beta.shape[0]]
                
                mean = numpy.dot(queryBetaPart, meanQueryPart)
                variance = queryBetaPart @ newCovMatrixQueryPart @ queryBetaPart.transpose()
                
            else:
                assert(len(sampleInfoTriple) == 2)
                queryCovariateIds, densityRegModel = sampleInfoTriple
                allMeans, allSTDs = densityRegModel.predict(observedCovariatesForClassifier, return_std=True)
                assert(allMeans.shape[0] == 1 and allSTDs.shape[0] == 1)
                mean = allMeans[0]
                variance = numpy.square(allSTDs[0])
                
                if numpy.isnan(mean) or numpy.isnan(variance):
                    print("! ERROR HERE ! ")
                    print("allMeans = ", allMeans)
                    print("allSTDs = ", allSTDs)
                    print("observedCovariatesForClassifier = ", observedCovariatesForClassifier)
                    print("densityRegModel = ", densityRegModel)
                    assert(False)
            
            assert((not numpy.isnan(mean)) and (not numpy.isnan(variance)))
            
            # if the variance is 0.0 then next value is deteriministic => no benifits of asking for that value 
            if variance > 0.0:
                if NUMBER_OF_SAMPLES is not None:
                    assert(False)
                    expectedFutureBayesRisk = getExpectedBayesRiskMCEstimateFast(nextMisclassifcationCosts, observedCovariates, mean, variance, nextPredictionModel, NUMBER_OF_SAMPLES)
                else:
                    expectedFutureBayesRisk = getExpectedFutureBayesRiskEstimate(nextMisclassifcationCosts, observedCovariates, mean, variance, nextPredictionModel)
                
                additionalFeatureCosts = numpy.sum(definedFeatureCosts[queryCovariateIds])
               
                # print("queryCovariateIds = " + str(queryCovariateIds) + ", expectedFutureBayesRisk = " + str(expectedFutureBayesRisk))
                totalExpectedFutureCosts = expectedFutureBayesRisk + additionalFeatureCosts  
                
                if totalExpectedFutureCosts < currentBayesRisk:
                    continueAskingForFeatures = True
                    break
            

            if onlyLookingOneStepAhead:
                break 
        
        
        if not continueAskingForFeatures:
            predictedProbs = currentPredictionModel.predict_proba(observedCovariatesForClassifier)
            predictedProbs = evaluation.ensure2D(predictedProbs)
            predictedLabel = evaluation.bayesClassifier(predictedProbs, currentMisclassificationCosts)
            
            assert(len(predictedLabel.shape) == 1 and predictedLabel.shape[0] == 1)
            return idsGivenCovariates, currentFeatureCosts, predictedLabel[0], predictedProbs[0,1]
    
    assert(False)
    return None



def runDynamicAcquisition_operationalCosts(sampleId, definedFeatureCosts, allFeatureSetsInOrder, allSamplerInfos, allPredictionModels, allThresholds, falsePositiveCost, testSample, onlyLookingOneStepAhead, NUMBER_OF_SAMPLES):
    assert(len(testSample.shape) == 1)
    assert(testSample.shape[0] == definedFeatureCosts.shape[0])
    assert(len(allFeatureSetsInOrder) == len(allSamplerInfos))
    assert(len(allFeatureSetsInOrder) == len(allPredictionModels))
    
    print("Process test sample ", sampleId)
    
    
    numberOfFeatureSets = len(allFeatureSetsInOrder)
    
    for i in range(numberOfFeatureSets):
        idsGivenCovariates = allFeatureSetsInOrder[i]
        observedCovariates = testSample[idsGivenCovariates]
        currentFeatureCosts = numpy.sum(definedFeatureCosts[idsGivenCovariates])
                
        currentPredictionModel = allPredictionModels[i]
        currentThreshold = allThresholds[i]
        
        observedCovariatesForClassifier = numpy.reshape(observedCovariates, (1, -1))  # classifier expects a 2D array
        predictedProbs = currentPredictionModel.predict_proba(observedCovariatesForClassifier)
        predictedProbs = evaluation.ensure2D(predictedProbs)
            
        assert(predictedProbs.shape[0] == 1)
        
        currentExpectedFalsePositiveCosts = evaluation.expectedFalsePositiveCosts_underThresholdRequirement(predictedProbs, currentThreshold, falsePositiveCost)
        # currentBayesRisk = evaluation.bayesRisk(predictedProbs, currentMisclassificationCosts)
                
        continueAskingForFeatures = False
        
        for j, sampleInfoTriple in enumerate(allSamplerInfos[i]):
            
            nextPredictionModel = allPredictionModels[i + j + 1]
            nextThreshold = allThresholds[i + j + 1]
            # nextMisclassifcationCosts = allMisclassificationCosts[i + j + 1]
            
            if len(sampleInfoTriple) == 3:
                
                assert(False) # because not valid anymore for GAM               
                queryCovariateIds, sigma12TimesSigma22InvQueryPart, newCovMatrixQueryPart = sampleInfoTriple
                assert(sigma12TimesSigma22InvQueryPart.shape[1] == observedCovariates.shape[0] and sigma12TimesSigma22InvQueryPart.shape[0] == queryCovariateIds.shape[0])
                meanQueryPart = numpy.matmul(sigma12TimesSigma22InvQueryPart, observedCovariates)
                
                assert(len(meanQueryPart.shape) == 1)
                assert(meanQueryPart.shape[0] > 0)
                assert(meanQueryPart.shape[0] == queryCovariateIds.shape[0])
                
                assert(sigma12TimesSigma22InvQueryPart.shape[0] > 0)
                if sigma12TimesSigma22InvQueryPart.shape[1] == 0:
                    assert(idsGivenCovariates.shape[0] == 0)
                    # this case means no given covariates
                    assert(numpy.all(meanQueryPart == 0))  # we assume that each covariate has zero mean
                assert(newCovMatrixQueryPart.shape[0] > 0 and newCovMatrixQueryPart.shape[1] > 0)
                  
                
                # print("allFeatureSetsInOrder[i + j + 1] = ", allFeatureSetsInOrder[i + j + 1])
                # print("numpy.append(idsGivenCovariates, queryCovariateIds) = ", numpy.append(idsGivenCovariates, queryCovariateIds))
                assert(numpy.all(numpy.equal(allFeatureSetsInOrder[i + j + 1], numpy.append(idsGivenCovariates, queryCovariateIds))))
                assert(nextPredictionModel.coef_.shape[1] == idsGivenCovariates.shape[0] + queryCovariateIds.shape[0])
                  
                beta = nextPredictionModel.coef_[0]
                queryBetaPart = beta[observedCovariates.shape[0]:beta.shape[0]]
                
                mean = numpy.dot(queryBetaPart, meanQueryPart)
                variance = queryBetaPart @ newCovMatrixQueryPart @ queryBetaPart.transpose()
                
            else:
                assert(len(sampleInfoTriple) == 2)
                queryCovariateIds, densityRegModel = sampleInfoTriple
                allMeans, allSTDs = densityRegModel.predict(observedCovariatesForClassifier, return_std=True)
                assert(allMeans.shape[0] == 1 and allSTDs.shape[0] == 1)
                mean = allMeans[0]
                variance = numpy.square(allSTDs[0])
                
                if numpy.isnan(mean) or numpy.isnan(variance):
                    print("! ERROR HERE ! ")
                    print("allMeans = ", allMeans)
                    print("allSTDs = ", allSTDs)
                    print("observedCovariatesForClassifier = ", observedCovariatesForClassifier)
                    print("densityRegModel = ", densityRegModel)
                    assert(False)
            
            assert((not numpy.isnan(mean)) and (not numpy.isnan(variance)))
            
            # if the variance is 0.0 then next value is deteriministic => no benifits of asking for that value 
            if variance > 0.0:
                if NUMBER_OF_SAMPLES is not None:
                    assert(False)
                else:
                    expectedFutureFalsePositiveCosts = getExpectedFutureFalsePositiveCostEstimate(nextThreshold, falsePositiveCost, observedCovariates, mean, variance, nextPredictionModel)
                    # expectedFutureBayesRisk = getExpectedBayesRiskEstimateProposed(nextMisclassifcationCosts, observedCovariates, mean, variance, nextPredictionModel)
                
                additionalFeatureCosts = numpy.sum(definedFeatureCosts[queryCovariateIds])
               
                # print("queryCovariateIds = " + str(queryCovariateIds) + ", expectedFutureBayesRisk = " + str(expectedFutureBayesRisk))
                # totalExpectedFutureCosts = expectedFutureBayesRisk + additionalFeatureCosts  
                totalExpectedFutureCosts = expectedFutureFalsePositiveCosts + additionalFeatureCosts
                
                if totalExpectedFutureCosts < currentExpectedFalsePositiveCosts:
                    continueAskingForFeatures = True
                    break
            

            if onlyLookingOneStepAhead:
                break 
        
        
        if not continueAskingForFeatures:
            predictedProbs = currentPredictionModel.predict_proba(observedCovariatesForClassifier)
            predictedProbs = evaluation.ensure2D(predictedProbs)
            # predictedLabel = evaluation.bayesClassifier(predictedProbs, currentMisclassificationCosts)
            predictedLabel = evaluation.thresholdClassifier(predictedProbs, currentThreshold)
            
            assert(len(predictedLabel.shape) == 1 and predictedLabel.shape[0] == 1)
            return idsGivenCovariates, currentFeatureCosts, predictedLabel[0], predictedProbs[0,1]
    
    assert(False)
    return None


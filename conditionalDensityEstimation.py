import prepareFeatureSets
import numpy
import dynamicAcquisition
import scipy.stats
import sklearn.gaussian_process
import sklearn.linear_model
import sklearn.neural_network

def evalHeldOutGP(GP, densityTestDataCovariates, densityTestDataResponse):
    allMeans, allSTDs = GP.predict(densityTestDataCovariates, return_std=True)
    
    mse = numpy.mean(numpy.square(allMeans - densityTestDataResponse))
    
    heldOutLogLikelihood = 0.0
    for t in range(densityTestDataCovariates.shape[0]):
        heldOutLogLikelihood += scipy.stats.norm.logpdf(x = densityTestDataResponse[t], loc = allMeans[t], scale = allSTDs[t])
        # print("mean = " + str(allMeans[t]) + ", std = " + str(allSTDs[t]))
    
    # assert(False)
    return heldOutLogLikelihood, mse
 

def evalHeldOutNormal(queryBetaPart, sigma12TimesSigma22InvQueryPart, newCovMatrixQueryPart, densityTestDataCovariates, densityTestDataResponse):
    heldOutLogLikelihoodNormal = 0.0
    mseNormal = 0.0
    for t in range(densityTestDataCovariates.shape[0]):
        observedCovariates = densityTestDataCovariates[t]
        meanQueryPart = numpy.matmul(sigma12TimesSigma22InvQueryPart, observedCovariates)
        mean = numpy.dot(queryBetaPart, meanQueryPart)
        variance = queryBetaPart @ newCovMatrixQueryPart @ queryBetaPart.transpose()
        heldOutLogLikelihoodNormal += scipy.stats.norm.logpdf(x = densityTestDataResponse[t], loc = mean, scale = numpy.sqrt(variance))
        mseNormal += numpy.square(mean - densityTestDataResponse[t])
     
    mseNormal = mseNormal / float(densityTestDataCovariates.shape[0])
     
    return heldOutLogLikelihoodNormal, mseNormal


# def GPwithLinearMeanModelOld(densityTrainDataCovariates, densityTrainDataResponse, densityTestDataCovariates, densityTestDataResponse, lengthScale, alphaValue, normalVariance):
# 
#     # train linear regression model
#     lrModel = sklearn.linear_model.LinearRegression()
#     lrModel.fit(densityTrainDataCovariates, densityTrainDataResponse)
#     
#     allLinearPredictionsTrain = lrModel.predict(densityTrainDataCovariates)
#     densityTrainDataResponseForGP = densityTrainDataResponse - allLinearPredictionsTrain
#      
#     covFunc = sklearn.gaussian_process.kernels.RBF(length_scale = lengthScale)
#     GP = sklearn.gaussian_process.GaussianProcessRegressor(kernel=covFunc, alpha=alphaValue, optimizer=None, normalize_y=False)
#     GP.fit(densityTrainDataCovariates, densityTrainDataResponseForGP)
#     
#     allLinearPredictionsTest = lrModel.predict(densityTestDataCovariates)
#     allGPPredictionsTest, allGPSTDsTest = GP.predict(densityTestDataCovariates, return_std=True)
#     finalTestPredictions = allGPPredictionsTest + allLinearPredictionsTest
#     
#     gpVariance = numpy.square(allGPSTDsTest)
#     
#     heldOutLogLikelihood = 0.0
#     for t in range(densityTestDataResponse.shape[0]):
#         heldOutLogLikelihood += scipy.stats.norm.logpdf(x = densityTestDataResponse[t], loc = finalTestPredictions[t], scale = numpy.sqrt(normalVariance + gpVariance[t]))
#     
#     mse = numpy.mean(numpy.square(finalTestPredictions - densityTestDataResponse))
#     print("mse (joint method) = ", mse)
#     print("log likelihood (joint method) = ", heldOutLogLikelihood)
#     
#     return


# mse (joint method) =  0.19895577446359233
# log likelihood (joint method) =  -276.5137049583004

def GPwithLinearMeanModel(densityTrainDataCovariates, densityTrainDataResponse, densityTestDataCovariates, densityTestDataResponse, lengthScale, alphaValue):

    # covFuncRBF = sklearn.gaussian_process.kernels.RBF(length_scale = 0.1)
    # covFuncDotProduct = sklearn.gaussian_process.kernels.DotProduct(sigma_0=10000.0)
    # covFuncFinal = sklearn.gaussian_process.kernels.Sum(covFuncRBF, covFuncDotProduct)
    # covFuncFinal = covFuncDotProduct 

    densityTrainDataCovariates = densityTrainDataCovariates[0:5000, ]
    densityTrainDataResponse = densityTrainDataResponse[0:5000, ]
    # print(densityTrainDataCovariates.shape[0])
    # assert(False)

    covFuncRBF = sklearn.gaussian_process.kernels.RBF(length_scale = 1.0)
    covFuncConst = sklearn.gaussian_process.kernels.ConstantKernel(constant_value=1.0)
    covFuncFinal = sklearn.gaussian_process.kernels.Product(covFuncConst, covFuncRBF)

    GP = sklearn.gaussian_process.GaussianProcessRegressor(kernel=covFuncFinal, alpha=1.0) # , optimizer=None, normalize_y=False)
    GP.fit(densityTrainDataCovariates, densityTrainDataResponse)

    allMeans, allSTDs = GP.predict(densityTestDataCovariates, return_std=True)
    
    mse = numpy.mean(numpy.square(allMeans - densityTestDataResponse))
    
    heldOutLogLikelihood = 0.0
    for t in range(densityTestDataCovariates.shape[0]):
        heldOutLogLikelihood += scipy.stats.norm.logpdf(x = densityTestDataResponse[t], loc = allMeans[t], scale = allSTDs[t])
        # print("allSTDs[t] = ", allSTDs[t])
        # heldOutLogLikelihood += scipy.stats.norm.logpdf(x = densityTestDataResponse[t], loc = allMeans[t], scale = numpy.sqrt(normalVariance))

    print("mse (joint method) = ", mse)
    print("log likelihood (joint method) = ", heldOutLogLikelihood)
    
    return heldOutLogLikelihood, mse


def BRModel(densityTrainDataCovariates, densityTrainDataResponse, densityTestDataCovariates, densityTestDataResponse, normalVariance):

    BR = sklearn.linear_model.BayesianRidge()
    BR.fit(densityTrainDataCovariates, densityTrainDataResponse)

    allMeans, allSTDs = BR.predict(densityTestDataCovariates, return_std=True)
    
    mse = numpy.mean(numpy.square(allMeans - densityTestDataResponse))
    
    heldOutLogLikelihood = 0.0
    for t in range(densityTestDataCovariates.shape[0]):
        heldOutLogLikelihood += scipy.stats.norm.logpdf(x = densityTestDataResponse[t], loc = allMeans[t], scale = allSTDs[t])
        print("allSTDs[t] = ", allSTDs[t])
        
    print("normalSTD = ", numpy.sqrt(normalVariance))
    print("mse (bayesian regression method) = ", mse)
    print("log likelihood (bayesian regression method) = ", heldOutLogLikelihood)
    
    return heldOutLogLikelihood, mse



def evaluateConditionalDensitySimple(trainDataFullCovariates, trainLabels, allData, testData):
    p = trainDataFullCovariates.shape[1]
    
    # idsGivenCovariates = numpy.arange(p-5)
    # idsNextCovariates = numpy.arange(p-4)
    
    idsGivenCovariates = numpy.arange(p-3)
    idsNextCovariates = numpy.arange(p-1)
    
    print("idsGivenCovariates = ", idsGivenCovariates)
    print("idsNextCovariates = ", idsNextCovariates)
    
    nextPredictionModel, _ = prepareFeatureSets.getOptimalTrainedModel(trainDataFullCovariates, trainLabels, idsNextCovariates)
        
    idsQueryCovariates = idsNextCovariates[idsGivenCovariates.shape[0] : idsNextCovariates.shape[0]]
    beta = nextPredictionModel.coef_[0]
    assert(beta.shape[0] == idsNextCovariates.shape[0])
    queryBetaPart = beta[idsGivenCovariates.shape[0] : idsNextCovariates.shape[0]]
    
    densityTrainDataCovariates = allData[:, idsGivenCovariates]
    densityTrainDataResponse = allData[:, idsQueryCovariates] @ queryBetaPart
    
    print("idsQueryCovariates = ", idsQueryCovariates)
    
    
    # train simple linear regression model
    sigma12TimesSigma22InvQueryPart, newCovMatrixQueryPart = dynamicAcquisition.getConditionalMeanAndVariance(allData, idsGivenCovariates, idsQueryCovariates)
    
    
    # print("densityTrainDataResponse average = ", numpy.average(densityTrainDataResponse))
    
    densityTestDataCovariates = testData[:, idsGivenCovariates]
    densityTestDataResponse = testData[:, idsQueryCovariates] @ queryBetaPart
    
    # train Gaussian process regression model
    # for lengthScale in [2.0, 1.0, 0.5]:
#     for lengthScale in [1.0]: # [10000.0, 100.0, 10.0, 1.0, 0.1, 0.001, 0.00001]:
#         for alphaValue in [1.0]: # [100.0, 10.0, 1.0, 0.1, 0.001]:
#             # covFunc = sklearn.gaussian_process.kernels.DotProduct(sigma_0 = lengthScale)
#             covFunc = sklearn.gaussian_process.kernels.RBF(length_scale = lengthScale)
#             GP = sklearn.gaussian_process.GaussianProcessRegressor(kernel=covFunc, alpha=alphaValue, optimizer=None, normalize_y=False)
#             # GP = sklearn.gaussian_process.GaussianProcessRegressor(kernel=sklearn.gaussian_process.kernels.RBF(), alpha=0.01, n_restarts_optimizer=5, normalize_y=False) 
#             GP.fit(densityTrainDataCovariates, densityTrainDataResponse)
#             # print("GP.kernel_.length_scale = ", GP.kernel_.length_scale)
#             print("lengthScale = " + str(lengthScale) + ", alphaValue = " + str(alphaValue))
#             # print("GP.log_marginal_likelihood_value_ = ", GP.log_marginal_likelihood_value_)
#             heldOutLogLikelihoodGP, mseGP = evalHeldOutGP(GP, densityTestDataCovariates, densityTestDataResponse)
#             # print("heldOutLogLikelihood (GP) = ", heldOutLogLikelihoodGP)
#             print("MSE (GP) = ", mseGP)
#             logLikelihoodTrainGP, mseTrainGP = evalHeldOutGP(GP, densityTrainDataCovariates, densityTrainDataResponse)
#             # print("logLikelihoodTrain (GP) = ", logLikelihoodTrainGP)
#             # print("mseTrain (GP)= ", mseTrainGP)
#     
    # train linear regression model
    # lrModel = sklearn.linear_model.LinearRegression()
#     for alphaValue in [100.0, 10.0, 1.0, 0.1, 0.001, 0.0001, 0.00001]:
#         print("alphaValue = ", alphaValue)
#         # lrModel = sklearn.linear_model.Ridge(alpha = alphaValue)
#         # lrModel.fit(densityTrainDataCovariates, densityTrainDataResponse)
#         
#         lrModel = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=100, alpha = alphaValue)
#         lrModel.fit(densityTrainDataCovariates, densityTrainDataResponse)
#         allMeansLR = lrModel.predict(densityTestDataCovariates)
#         mseLR = numpy.mean(numpy.square(allMeansLR - densityTestDataResponse))
#         print("MSE (linear regression) = ", mseLR)

    normalVariance = queryBetaPart @ newCovMatrixQueryPart @ queryBetaPart.transpose()
    
    lengthScale = 2.0 
    alphaValue = 100000.0 # 0.00001  corresponds to 1/sigma_noise
    # alphaValue = 1.0 # 1.0 / normalVariance
    
    BRModel(densityTrainDataCovariates, densityTrainDataResponse, densityTestDataCovariates, densityTestDataResponse, normalVariance)
    GPwithLinearMeanModel(densityTrainDataCovariates, densityTrainDataResponse, densityTestDataCovariates, densityTestDataResponse, lengthScale, alphaValue)
   
    heldOutLogLikelihoodNormal, mseNormal = evalHeldOutNormal(queryBetaPart, sigma12TimesSigma22InvQueryPart, newCovMatrixQueryPart, densityTestDataCovariates, densityTestDataResponse)
    print("heldOutLogLikelihoodNormal = ", heldOutLogLikelihoodNormal)
    print("mseNormal = ", mseNormal)
    
    logLikelihoodTrainNormal, mseTrainNormal = evalHeldOutNormal(queryBetaPart, sigma12TimesSigma22InvQueryPart, newCovMatrixQueryPart, densityTrainDataCovariates, densityTrainDataResponse)
    print("logLikelihoodTrainNormal = ", logLikelihoodTrainNormal)
    print("mseTrainNormal = ", mseTrainNormal)
    
    print("numpy.mean(densityTestDataResponse) = ", numpy.mean(densityTestDataResponse))
    print("mse of zero estimate = ", numpy.mean(numpy.square(densityTestDataResponse)))
    return

    
def evaluateConditionalDensity(trainDataFullCovariates, trainLabels, allData, allFeatureListsInOrder, testData):
    
    allPredictionModels = []
    numberOfFeatureSets = len(allFeatureListsInOrder)
        
    for i in range(numberOfFeatureSets):
        idsGivenCovariates = allFeatureListsInOrder[i]
        predictionModel, _ = prepareFeatureSets.getOptimalTrainedModel(trainDataFullCovariates, trainLabels, idsGivenCovariates)
        allPredictionModels.append(predictionModel)


    for i in range(numberOfFeatureSets):
        idsGivenCovariates = allFeatureListsInOrder[i]
        
        if idsGivenCovariates.shape[0] != 6:
            continue
        
        print("---------------------------")
        print("idsGivenCovariates = ", idsGivenCovariates)
        for j in range(i + 1, numberOfFeatureSets):
            nextPredictionModel = allPredictionModels[j]
            idsNextCovariates = allFeatureListsInOrder[j]
            assert((trainDataFullCovariates[:, idsNextCovariates]).shape[1] > 0)
            
           
            idsQueryCovariates = idsNextCovariates[idsGivenCovariates.shape[0] : idsNextCovariates.shape[0]]
            beta = nextPredictionModel.coef_[0]
            assert(beta.shape[0] == idsNextCovariates.shape[0])
            queryBetaPart = beta[idsGivenCovariates.shape[0] : idsNextCovariates.shape[0]]
            
            print("idsQueryCovariates = ", idsQueryCovariates)
            
            
            # train simple linear regression model
            sigma12TimesSigma22InvQueryPart, newCovMatrixQueryPart = dynamicAcquisition.getConditionalMeanAndVariance(allData, idsGivenCovariates, idsQueryCovariates)
            
            densityTrainDataCovariates = allData[:, idsGivenCovariates]
            densityTrainDataResponse = allData[:, idsQueryCovariates] @ queryBetaPart
            
            # print("densityTrainDataResponse average = ", numpy.average(densityTrainDataResponse))
            
            densityTestDataCovariates = testData[:, idsGivenCovariates]
            densityTestDataResponse = testData[:, idsQueryCovariates] @ queryBetaPart
            
            # train Gaussian process regression model
            # for lengthScale in [2.0, 1.0, 0.5]:
            for lengthScale in [100.0, 10.0, 1.0, 0.1, 0.001]:
                for alphaValue in [20.0, 10.0, 5.0]:  
                    covFunc = sklearn.gaussian_process.kernels.DotProduct(sigma_0 = lengthScale)
                    # covFunc = sklearn.gaussian_process.kernels.RBF(length_scale = lengthScale)
                    GP = sklearn.gaussian_process.GaussianProcessRegressor(kernel=covFunc, alpha=alphaValue, optimizer=None, normalize_y=False)
                    # GP = sklearn.gaussian_process.GaussianProcessRegressor(kernel=sklearn.gaussian_process.kernels.RBF(), alpha=0.01, n_restarts_optimizer=5, normalize_y=False) 
                    GP.fit(densityTrainDataCovariates, densityTrainDataResponse)
                    # print("GP.kernel_.length_scale = ", GP.kernel_.length_scale)
                    print("lengthScale = " + str(lengthScale) + ", alphaValue = " + str(alphaValue))
                    print("GP.log_marginal_likelihood_value_ = ", GP.log_marginal_likelihood_value_)
                    heldOutLogLikelihoodGP, mseGP = evalHeldOutGP(GP, densityTestDataCovariates, densityTestDataResponse)
                    print("heldOutLogLikelihoodGP = ", heldOutLogLikelihoodGP)
            
             
            heldOutLogLikelihoodNormal = 0.0
            mseNormal = 0.0
            for t in range(densityTestDataCovariates.shape[0]):
                observedCovariates = densityTestDataCovariates[t]
                meanQueryPart = numpy.matmul(sigma12TimesSigma22InvQueryPart, observedCovariates)
                mean = numpy.dot(queryBetaPart, meanQueryPart)
                variance = queryBetaPart @ newCovMatrixQueryPart @ queryBetaPart.transpose()
                heldOutLogLikelihoodNormal += scipy.stats.norm.logpdf(x = densityTestDataResponse[t], loc = mean, scale = numpy.sqrt(variance))
                mseNormal += numpy.square(mean - densityTestDataResponse[t])
             
            mseNormal = mseNormal / float(densityTestDataCovariates.shape[0])
            print("heldOutLogLikelihoodNormal = ", heldOutLogLikelihoodNormal)
            print("mseNormal = ", mseNormal)
            
            
            
        
       
    return 

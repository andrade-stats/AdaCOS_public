import numpy


# def estimateFullCovMatrix(allData):
#     covMatrix = numpy.cov(allData.transpose())
#     return covMatrix




def estimateFullCovMatrix_highDim(data):
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    
    importr("FastImputation")
    r = robjects.r
    
    robjects.globalenv["data"] = data
    covarianceMatrixEstimate = r('''CovarianceWithMissing(data)''')
    
    return covarianceMatrixEstimate


def estimateFullCovMatrix_mvnmle(data):
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    
    importr("mvnmle")
    r = robjects.r
    
    # r('data(apple)')
    robjects.globalenv["data"] = data
    # robjects.globalenv["data"] = r('apple')
    # print("my data = ")
    # print(robjects.globalenv["data"])
    
    covarianceMatrixEstimate = r('''mlest(data)$sigmahat''')
    
    # print("covarianceMatrixEstimate = ")
    # print(covarianceMatrixEstimate)
    return covarianceMatrixEstimate
    
#     r('''myImputerForData <- mice(data, m = 5, method = 'pmm', seed = 101)''')
#     
#     allDataImputed = []
#     for i in range(1,nrImputedDataSets+1):
#         imputedData = r('data.matrix(complete(myImputerForData,' + str(i) + '))')
#         # for the remaining NAN use simple mean imputation
#         imputedData = meanImputation(imputedData)
#         assert(not numpy.any(numpy.isnan(imputedData)))
#         allDataImputed.append(imputedData)
#     
#     return allDataImputed



def getMissingMeanAndCov(fullCovMatrix, covariateVector):
    assert(fullCovMatrix.shape[0] == covariateVector.shape[0])
    
    idsMissingCovariates = numpy.where(numpy.isnan(covariateVector))[0]
    idsGivenCovariates  = numpy.where(numpy.logical_not(numpy.isnan(covariateVector)))[0]
    
    # print("covariateVector = ")
    # print(covariateVector)
    # print("idsMissingCovariates = ", idsMissingCovariates)
    # print("idsGivenCovariates = ", idsGivenCovariates)
    # assert(False)
    
    p = fullCovMatrix.shape[0]
    assert(idsMissingCovariates.shape[0] >= 1 and idsGivenCovariates.shape[0] >= 1)
    assert((idsMissingCovariates.shape[0] + idsGivenCovariates.shape[0]) == p)
    
    newOrder = numpy.hstack((idsMissingCovariates, idsGivenCovariates))
    assert(newOrder.shape[0] == p)
    
    covMatrix = numpy.copy(fullCovMatrix)
    covMatrix = covMatrix[:, newOrder]
    covMatrix = covMatrix[newOrder, :]
    
    # print("covMatrix = ")
    # print(covMatrix)
    # assert(False)
    
    upperLeftBlockSize = idsMissingCovariates.shape[0]
    
    upperRightBlock = covMatrix[0:upperLeftBlockSize, upperLeftBlockSize:p]
    upperLeftBlock = covMatrix[0:upperLeftBlockSize, 0:upperLeftBlockSize]
    lowerRightBlock = covMatrix[upperLeftBlockSize:p, upperLeftBlockSize:p]
    
    # relevant matrices for the characterization of the normal distribution of rest variables given "idsGivenCovariates" variables
    sigma12TimesSigma22Inv = numpy.matmul(upperRightBlock, numpy.linalg.inv(lowerRightBlock + 0.01 * numpy.eye(lowerRightBlock.shape[0])))
    missingCovMatrix = upperLeftBlock - numpy.matmul(sigma12TimesSigma22Inv, upperRightBlock.transpose())
    
    observedCovariates = covariateVector[idsGivenCovariates]
    missingMean = numpy.matmul(sigma12TimesSigma22Inv, observedCovariates)
    
    return missingMean, missingCovMatrix


def imputeData(incompleteData):
    assert(numpy.any(numpy.isnan(incompleteData)))
    
    fullCovMatrix = estimateFullCovMatrix_highDim(incompleteData)
    # fullCovMatrix = estimateFullCovMatrix_mvnmle(incompleteData)
    
    imputedData = numpy.copy(incompleteData)
    
    for i in range(imputedData.shape[0]):
        if numpy.sum(numpy.isnan(incompleteData[i])) >= 1:
            estimatedMean, estimatedCovMatrix = getMissingMeanAndCov(fullCovMatrix, incompleteData[i])
            imputedData[i, numpy.isnan(incompleteData[i])] = estimatedMean
    
    assert(not numpy.any(numpy.isnan(imputedData)))
    return imputedData
    
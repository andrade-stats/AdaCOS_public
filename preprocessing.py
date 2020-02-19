
import numpy

def standardizeDataWithMeanStd(allMeans, allStds, designMatrix):
    standardizedDesignMatrix = designMatrix - allMeans
    standardizedDesignMatrix = standardizedDesignMatrix / allStds
    return standardizedDesignMatrix


def standardizeAllDataNew(trainData, unlabeledData, testData):
    
    # use all data
    allData = numpy.vstack((trainData, unlabeledData))
    allMeans = numpy.nanmean(allData, axis = 0)
    allStds = numpy.nanstd(allData, axis = 0)
    
    # print("allMeans = ", allMeans)
    # print("allStds = ", allStds)
    
    # ensure that all stds > 0
    allStds[allStds == 0] = 1.0
    
    assert(not numpy.any(numpy.isnan(allMeans)))
    assert(not numpy.any(numpy.isnan(allStds)))
    assert(numpy.all(allStds) > 0.0)
    
    trainData = standardizeDataWithMeanStd(allMeans, allStds, trainData)
    unlabeledData = standardizeDataWithMeanStd(allMeans, allStds, unlabeledData)
    testData = standardizeDataWithMeanStd(allMeans, allStds, testData)
    return trainData, unlabeledData, testData


# assume that each row is one sample
def meanImputation(data):
    
    allMeans = numpy.nanmean(data, axis = 0)
    assert(not numpy.any(numpy.isnan(allMeans)))
    
    imputedData = numpy.copy(data)
    
    for i in range(data.shape[0]):
        nanPositions = numpy.where(numpy.isnan(data[i]))[0]
        imputedData[i, nanPositions] = allMeans[nanPositions]

    return imputedData


def multipleImputationMethod(data, nrImputedDataSets = 5):
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    
    importr("mice")
    r = robjects.r
    
    robjects.globalenv["data"] = data
    r('''myImputerForData <- mice(data, m = 5, method = 'pmm', seed = 101)''')
    
    allDataImputed = []
    for i in range(1,nrImputedDataSets+1):
        imputedData = r('data.matrix(complete(myImputerForData,' + str(i) + '))')
        # for the remaining NAN use simple mean imputation
        imputedData = meanImputation(imputedData)
        assert(not numpy.any(numpy.isnan(imputedData)))
        allDataImputed.append(imputedData)
    
    return allDataImputed


import numpy
import realdata
import experimentHelper
from fancyimpute import NuclearNormMinimization
import preprocessing
import constants
import pickle
import gaussianImputation

imputationMethod = "gaussian_imputation"

# imputationMethod = "mice_imputation_all"
# imputationMethod = "nuclearNorm_imputation"
# imputationMethod = "mean_imputation"

# dataName = "pyhsioNetWithMissing_5foldCV"

dataName = "heartDiseaseWithMissing_5foldCV"
     
for foldId in range(5):
    trainData, _, _, _, _ = realdata.loadSubsetBasic(dataName, None, foldId)
    
    stemFilenameForData = dataName + "_" + imputationMethod
    filename = constants.BASE_FOLDER + stemFilenameForData + "_fold" + str(foldId) + "_trainData"
    
    if imputationMethod == "mean_imputation":
        print("start mean imputation for fold ", foldId)
        imputedTrainingData = preprocessing.meanImputation(trainData)
        assert(not numpy.any(numpy.isnan(imputedTrainingData)))
        numpy.save(filename, imputedTrainingData)
    elif imputationMethod == "nuclearNorm_imputation":
        print("start nuclear norm minimization for fold ", foldId)
        imputedTrainingData = NuclearNormMinimization().fit_transform(trainData)
        assert(not numpy.any(numpy.isnan(imputedTrainingData)))
        numpy.save(filename, imputedTrainingData)
        
    elif imputationMethod == "gaussian_imputation":
        print("start gaussian imputation for fold ", foldId)
        imputedTrainingData = gaussianImputation.imputeData(trainData)
        numpy.save(filename, imputedTrainingData)
    elif imputationMethod == "mice_imputation_all":
        allImputedData = preprocessing.multipleImputationMethod(trainData)
        # print("nan in training data = ", numpy.count_nonzero(numpy.isnan(trainData)))
        # print("nan in imputeed 1 training data = ", numpy.count_nonzero(numpy.isnan(imputedData)))
        # imputedData = preprocessing.meanImputation(imputedData)
        # print("nan in imputed 2 training data = ", numpy.count_nonzero(numpy.isnan(imputedData)))
        
        with open(filename, 'wb') as f:
            pickle.dump(allImputedData, f)
    else:
        assert(False)   

print("FINISHED ALL")
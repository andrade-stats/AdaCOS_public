import numpy
import realdata
import sklearn.metrics
import evaluation
import experimentHelper
import dynamicAcquisition
import time
import sys
import prepareFeatureSets
from multiprocessing import Pool
import preprocessing
import constants
import sklearn.linear_model
import sklearn.gaussian_process
import scipy.stats
import MyLayers
import tensorflow as tf

# dataName = "pima_5foldCV"
dataName = "pyhsioNetWithMissing_5foldCV"

# densityRegressionModelName = "BR"
densityRegressionModelName = "GP"
OBSERVED_COVARIATES_NR = 3 # try out 1 and 2
foldId = 0

# TRAINING PREDICTIONS:
# logProb =  -15167.074498208389
# mse =  0.8664656307034838
# HELD-OUT PREDICTIONS:
# logProb =  -1050.59030358889
# mse =  0.9856081035014143
# HELD-OUT BASELINE:
# logProb =  -1077.5471768498187
# mse =  1.0665627633745327


imputationMethod = "gaussian_imputation"

definedFeatureCosts = realdata.getFeaturesCosts(dataName)

trainData, trainLabels, unlabeledData, testData, testLabels = realdata.loadSubset(dataName, None, foldId, imputationMethod)
    
allFeatureSetsInOrder, _ = prepareFeatureSets.getAllFeatureSetsInOrderWithL1LogReg(trainData, trainLabels, unlabeledData, None, definedFeatureCosts)

observedIds = allFeatureSetsInOrder[OBSERVED_COVARIATES_NR]
predictedId = numpy.sort(list(set(allFeatureSetsInOrder[OBSERVED_COVARIATES_NR+1]) - set(observedIds)))
predictedId = list(predictedId)[0]
print("predictedId = ", predictedId)


densityTrainDataCovariates = trainData[:,observedIds]
densityTrainDataResponse = trainData[:,predictedId]

densityTestDataCovariates = testData[:,observedIds]
densityTestDataResponse = testData[:,predictedId]

print("observedIds = ", observedIds)
print("predictedId = ", predictedId)



if densityRegressionModelName == "BR":
    model = sklearn.linear_model.BayesianRidge()
elif densityRegressionModelName == "GP":
    covFuncRBF = sklearn.gaussian_process.kernels.RBF(length_scale = 0.1) # 0.0001)
    # covFuncConst = sklearn.gaussian_process.kernels.ConstantKernel(constant_value=1.0)
    # covFuncFinal = sklearn.gaussian_process.kernels.Product(covFuncConst, covFuncRBF)
    # model = sklearn.gaussian_process.GaussianProcessRegressor(kernel=covFuncFinal, alpha=1.0) # , optimizer=None, normalize_y=False)
    model = sklearn.gaussian_process.GaussianProcessRegressor(kernel=covFuncRBF, alpha = 0.1, optimizer=None) # , normalize_y=False)
    # model = sklearn.gaussian_process.GaussianProcessRegressor(kernel=covFuncRBF, alpha = 0.1) # , normalize_y=False)
    
elif densityRegressionModelName == "NN":
    
    TRANSFORMATION_WEIGHT_REGULARIZER = 0.0
    
    # Build model.
    model = tf.keras.Sequential([
        # MyLayers.DuplicateInput(10, TRANSFORMATION_WEIGHT_REGULARIZER),
        tf.keras.layers.Dense(100, activation=tf.keras.activations.sigmoid),
        tf.keras.layers.Dense(1, activation=None, use_bias=False)
    ])
    
    
    # Do inference.
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.01), loss="mse")
    model.fit(densityTrainDataCovariates, densityTrainDataResponse, epochs=10000, verbose=True) # batch_size=10)




if densityRegressionModelName == "NN":
    print("TRAINING PREDICTIONS:")
    predicted_means = model.predict(densityTrainDataCovariates)[:,0]
    evaluation.showPredctions(predicted_means, None, densityTrainDataResponse)
    
    # print("predicted_means = ")
    # print(predicted_means)
    
    print("HELD-OUT PREDICTIONS:")
    predicted_means = model.predict(densityTestDataCovariates)[:,0]
    evaluation.showPredctions(predicted_means, None, densityTestDataResponse)
    
    
else:

    model.fit(densityTrainDataCovariates, densityTrainDataResponse)

    print("TRAINING PREDICTIONS:")
    predicted_means, allSTDs = model.predict(densityTrainDataCovariates, return_std=True)
    evaluation.showPredctions(predicted_means, allSTDs, densityTrainDataResponse)
    
    print("HELD-OUT PREDICTIONS:")
    predicted_means, allSTDs = model.predict(densityTestDataCovariates, return_std=True)
    # print("predicted_means = ", predicted_means)
    evaluation.showPredctions(predicted_means, allSTDs, densityTestDataResponse)



print("HELD-OUT BASELINE:")
predicted_means, allSTDs = numpy.zeros(densityTestDataCovariates.shape[0]), numpy.ones(densityTestDataCovariates.shape[0]) 
evaluation.showPredctions(predicted_means, allSTDs, densityTestDataResponse)



# logProb =  2667.5334444399095
# mse =  0.007798541032153731
# HELD-OUT PREDICTIONS:
# logProb =  -1108.0511724458938
# mse =  1.0730835790452726
# HELD-OUT BASELINE:
# logProb =  -1077.5471768498187
# mse =  1.0665627633745327


import numpy
# import realdata
# import sklearn.linear_model
# import sklearn.metrics
# import evaluation
# import experimentSetting
# import experimentHelper
# import prepareFeatureSets
# import sklearn.gaussian_process
# import time
# import constants
# import preprocessing
# import pickle
# from pygam import LogisticGAM


# import tensorflow_probability as tfp
# import scipy.stats

import time
import runClassifier
import sys
import evaluation
from multiprocessing import Pool

# /opt/intel/intelpython3/bin/python evalClassificationAccuracyOnly_onlyNN.py pyhsioNetWithMissing_5foldCV
# /opt/intel/intelpython3/bin/python evalClassificationAccuracyOnly_onlyNN.py breastcancer_5foldCV
# /opt/intel/intelpython3/bin/python evalClassificationAccuracyOnly_onlyNN.py pima_5foldCV
# /opt/intel/intelpython3/bin/python evalClassificationAccuracyOnly_onlyNN.py heartDiseaseWithMissing_5foldCV

# dataName = "pima_5foldCV"
# dataName = "breastcancer_5foldCV"
# dataName = "pyhsioNetWithMissing_5foldCV"
# dataName = "heartDiseaseWithMissing_5foldCV"

# EPOCHS = 1000
# BATCH_SIZE=100
# NUMBER_OF_CV_FOLDS = 10
# ALL_WEIGHT_REG_CANDIDATES = [100.0, 10.0, 1.0, 0.1, 0.01, 0.001]


if len(sys.argv) >= 2:
    dataName = sys.argv[1]
else:
    
    # dataName = "miniBooNE_5foldCV"
    # dataName = "crab_5foldCV"
    # dataName = "pima_5foldCV"
    dataName = "breastcancer_5foldCV"
    
    # dataName = "pyhsioNetNoMissing_5foldCV"
    # dataName = "pyhsioNetWithMissing_5foldCV"

imputationMethod = "gaussian_imputation"

NUMBER_OF_FOLDS = 5

# /opt/intel/intelpython3/bin/python evalClassificationAccuracyOnly_onlyGAM.py

# NN (lambda = 1.0)
# breastcancer_5foldCV: auc = 0.9956 (0.0039)
# pima_5foldCV: auc = 0.8243 (0.0256)

# logistic regression
# breastcancer_5foldCV: auc = 0.995 (0.0048)
# pima_5foldCV: auc = 0.8257 (0.023
# crab_5foldCV: auc = 0.9965 (0.0044)
# miniBooNE_5foldCV: auc = 0.9318 (0.0249)
# pyhsioNetWithMissing_5foldCV: auc = 0.8421 (0.0158)


# classWeightRegularizer = 0.1
# TRAIN DATA:
# auc =  0.8381308411214954
# logLikelihood =  -0.49227598
# TEST DATA:
# auc =  0.8627777777777779
# logLikelihood =  -0.47817636

# classWeightRegularizer = 0.01
# TRAIN DATA:
# auc =  0.8589369158878505
# logLikelihood =  -0.44515565
# TEST DATA:
# auc =  0.8735185185185186
# logLikelihood =  -0.43821388

# TRANSFORMATION_WEIGHT_REGULARIZER = 1.0
# ALL_CLASS_WEIGHT_REGULARIZER = [0.01, 0.1, 1.0, 10.0]
# ALL_CLASS_WEIGHT_REGULARIZER = [0.01, 0.1]




# AVERAGE RESULTS 5-FOLD CROSS-VAlIDATION
# *************************** AVERAGE ACCURACY OVER ALL FOLDS  *******************************************
# dataName =  pima_5foldCV
# *************************** NN *******************************************
# auc = 0.8372 (0.0206)


numpy.random.seed(3523421)

# train auc =  0.8582827102803738
# test auc =  0.8738888888888889

# classWeightRegularizer = 0.1
# train auc =  0.8272429906542057
# test auc =  0.8646296296296296


 
    

# testAUCAllFoldsNN = numpy.zeros(NUMBER_OF_FOLDS)

# runClassifier.run(dataName, 0, imputationMethod)
# assert(False)

# runClassifier.play(dataName, 0, imputationMethod)


proposedMethod = True

startTimeTotal = time.time()

allParamsForMultiprocessMap = []
for foldId in range(NUMBER_OF_FOLDS):
    allParamsForMultiprocessMap.append((dataName, foldId, imputationMethod, proposedMethod))

with Pool(NUMBER_OF_FOLDS) as pool:
    allResults = pool.starmap(runClassifier.run, allParamsForMultiprocessMap)

print("FINISHED")
print("total runtime (in minutes) = " + str((time.time() - startTimeTotal) / 60.0))

allResults = numpy.asarray(allResults)

testLogLikelihood = allResults[:,0]
evalLogLikelihood = allResults[:,1]
trainLogLikelihood = allResults[:,2]
testAUCAllFolds = allResults[:,3]
bestWeightParams = allResults[:,4]
bestTransformParam = allResults[:,5]
 
print("SUMMARY FOR DEVELOPMENT: ")
print("proposedMethod = ", proposedMethod)
print("average train = ", numpy.mean(trainLogLikelihood))
print("average eval = ", numpy.mean(evalLogLikelihood))
print("average test = ", numpy.mean(testLogLikelihood))
print("bestWeightParams:")
print(bestWeightParams)

print("bestTransformParam:")
print(bestTransformParam)

print("")
print("AVERAGE RESULTS 5-FOLD CROSS-VAlIDATION")

print("*************************** AVERAGE ACCURACY OVER ALL FOLDS  *******************************************")
print("dataName = ", dataName)

print("*************************** NN *******************************************")
evaluation.showHelperDetailed("auc = ", testAUCAllFolds)



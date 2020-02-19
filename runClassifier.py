
import sklearn.model_selection
import pandas


import MyLayers    
import tensorflow as tf

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import evaluation
import realdata
import numpy


def createModelProposed(transformWeightRegularizer=0.0, classWeightRegularizer=0.0, nrTransformationUnits = 10, learningRate = 0.01):
    
    model = tf.keras.Sequential([
                    MyLayers.DuplicateInput_withSkip(nrTransformationUnits, transformWeightRegularizer),
                    tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer = tf.keras.regularizers.l2(classWeightRegularizer), use_bias=True)
                    ])
  
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=learningRate), loss="binary_crossentropy")
    return model
    

def createModelLogReg(transformWeightRegularizer=None, classWeightRegularizer=0.0, nrTransformationUnits = None, learningRate = 0.01):
    
    model = tf.keras.Sequential([
                    tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer = tf.keras.regularizers.l2(classWeightRegularizer), use_bias=True)
                    ])
  
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=learningRate), loss="binary_crossentropy")
    return model
    
def myScorer(estimator, X_eval, y_eval):
    predictedProbs = estimator.predict_proba(X_eval)
    logLikelihood = evaluation.getAverageHeldOutLogLikelihood_FromProbabilities(predictedProbs, y_eval)
    return logLikelihood



# TRAIN DATA:
# auc =  0.9948971314799793
# logLikelihood =  -0.414529
# TEST DATA:
# auc =  0.985252808988764
# logLikelihood =  -0.41399974

# TRAIN DATA:
# auc =  0.9969323796180223
# logLikelihood =  -0.069488354
# TEST DATA:
# auc =  0.9936797752808988
# logLikelihood =  -0.11151198

def play(dataName, foldId, imputationMethod):
    EPOCHS = 1000
    # BATCH_SIZE=100
    
    createModel = createModelProposed
    # createModel = createModelLogReg
    
    trainData, trainLabels, unlabeledData, testData, testLabels = realdata.loadSubset(dataName, None, foldId, imputationMethod)
    
    
    finalModel = createModel(transformWeightRegularizer=0.1, classWeightRegularizer=0.001, nrTransformationUnits = 10, learningRate = 0.01)
    finalModel.fit(trainData, trainLabels, epochs=EPOCHS, verbose=True)
    aucTest, logLikelihoodTest = evaluation.eval_NN(finalModel, testData, testLabels)

    print("TRAIN DATA:")
    aucTrain, logLikelihoodTrain = evaluation.eval_NN(finalModel, trainData, trainLabels)
    print("auc = ", aucTrain)
    print("logLikelihood = ", logLikelihoodTrain)
         
    print("TEST DATA:")
    print("auc = ", aucTest)
    print("logLikelihood = ", logLikelihoodTest)
    assert(False)
    return
     

def run(dataName, foldId, imputationMethod, proposedMethod):
    
    EPOCHS = 2000
    NUMBER_OF_CV_FOLDS = 10
    ALL_WEIGHT_REG_CANDIDATES = [1.0, 0.1, 0.01, 0.001, 0.0001]
    ALL_TRANSFORM_REG_CANDIDATES = [1.0, 0.1, 0.01, 0.001, 0.0]
    
    # ALL_WEIGHT_REG_CANDIDATES = [1.0]
    # ALL_TRANSFORM_REG_CANDIDATES = [1.0]
    
    NR_JOBS = 1


    if proposedMethod:
        createModel = createModelProposed
    else:
        createModel = createModelLogReg
    
    modelForCV = KerasClassifier(build_fn=createModel, epochs=EPOCHS, verbose=True)
    
    trainData, trainLabels, unlabeledData, testData, testLabels = realdata.loadSubset(dataName, None, foldId, imputationMethod)

    parameters = {"classWeightRegularizer" : ALL_WEIGHT_REG_CANDIDATES, "transformWeightRegularizer" : ALL_TRANSFORM_REG_CANDIDATES}
    gridSearchObj = sklearn.model_selection.GridSearchCV(modelForCV, parameters, scoring = myScorer, cv = NUMBER_OF_CV_FOLDS, n_jobs = NR_JOBS)
    gridSearchObj.fit(trainData, trainLabels)
    
    cvResult = pandas.DataFrame.from_dict(gridSearchObj.cv_results_)
    meanScoresEval = (cvResult["mean_test_score"]).as_matrix()
    bestId = numpy.argmax(meanScoresEval)
    
    bestWeightParam = cvResult.loc[bestId, "param_classWeightRegularizer"]
    bestTransformParam = cvResult.loc[bestId, "param_transformWeightRegularizer"]
    meanScoresTrain = (cvResult["mean_train_score"]).as_matrix()
    
    
    finalModel = createModel(transformWeightRegularizer=bestTransformParam, classWeightRegularizer=bestWeightParam)
    finalModel.fit(trainData, trainLabels, epochs=EPOCHS, verbose=True)
    aucTest, logLikelihood = evaluation.eval_NN(finalModel, testData, testLabels)
    
#     print("bestWeightParam = ")
#     print(bestWeightParam)
#     print("meanScores = ")
#     print(meanScores)
#     print("TRAIN DATA:")
#     auc, logLikelihood = evaluation.eval_NN(finalModel, trainData, trainLabels)
#     print("auc = ", auc)
#     print("logLikelihood = ", logLikelihood)
         
        
    # print("TEST DATA:")
    # print("auc = ", aucTest)
    # print("logLikelihood = ", logLikelihood)
    
#     print("average training score = ", meanScoresTrain[bestId])
#     print("average eval score = ", meanScoresEval[bestId])
#     print("test score = ", logLikelihood)
    
    return logLikelihood, meanScoresEval[bestId], meanScoresTrain[bestId], aucTest, bestWeightParam, bestTransformParam
 
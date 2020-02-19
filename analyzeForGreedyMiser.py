import numpy
import experimentSetting
import experimentHelper
import evaluation
import scipy.io
import os


# dataName = "pima_5foldCV"
# dataName = "miniBooNE_5foldCV"
dataName = "breastcancer_5foldCV"

if dataName == "pima_5foldCV":
    sameClassCost = -50.0 # set to -50.0 in order to compare to Li and Carin work
    # assert(misclassificationCostsSymmetric == 400 or misclassificationCostsSymmetric == 800)
else:
    sameClassCost = 0.0 
    



NUMBER_OF_FOLDS = 5

allMisclassificationCostsSymmetric = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]

resultsRecorder = experimentHelper.ResultsRecorder(len(allMisclassificationCostsSymmetric))
readyForSaving = False

for misclassificationCostsSymmetric in allMisclassificationCostsSymmetric:
    
    validTotalCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    validFeatureCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    validMisClassificationCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    validAccuracyAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
    
    allTestFoldAvgBestTotalCostsValidResult = numpy.zeros(NUMBER_OF_FOLDS) # for analysis only
     
    allBestSettings = numpy.zeros((NUMBER_OF_FOLDS, 2), dtype = numpy.int)
     
    for testFoldId in range(NUMBER_OF_FOLDS):
    
        print("*************************** GREEDY MISER *******************************************")
         
        totalCostsValidResult, validAccuracy, validFeatureCosts, bestLambdaId, bestTreeId, avgBestTotalCostsValidResult = evaluation.getBestAverage10TrainFoldTotalCostsResultSymmetricForGreedyMiser(dataName, testFoldId, misclassificationCostsSymmetric, sameClassCost)
        allTestFoldAvgBestTotalCostsValidResult[testFoldId] = avgBestTotalCostsValidResult
        allBestSettings[testFoldId, 0] = bestLambdaId
        allBestSettings[testFoldId, 1] = bestTreeId
        
        validTotalCostsAllFolds[testFoldId] = totalCostsValidResult
        validFeatureCostsAllFolds[testFoldId] = validFeatureCosts
        validMisClassificationCostsAllFolds[testFoldId] = (1.0 - validAccuracy) * misclassificationCostsSymmetric + validAccuracy * sameClassCost
        validAccuracyAllFolds[testFoldId] = validAccuracy
        # print("bestLambdaId = ", bestLambdaId)
    
    print("*************************** AVERAGE OVER ALL FOLDS (misclassification costs = " + str(misclassificationCostsSymmetric) + ", sameClassCost = " + str(sameClassCost) + ") *******************************************")
    evaluation.showHelper("total costs = ", validTotalCostsAllFolds)
    evaluation.showHelper("feature costs = ", validFeatureCostsAllFolds)
    evaluation.showHelper("misclassification costs = ", validMisClassificationCostsAllFolds)
    evaluation.showHelper("accuracy = ", validAccuracyAllFolds)
    
    evaluation.showHelper("average best validation costs (for analysis only) = ", allTestFoldAvgBestTotalCostsValidResult)
    
    # save best settings from validation data
    outputFilename = "/export/home/s-andrade/dynamicCovariateBaselines/GreedyMiser/" + dataName + "_" + str(int(misclassificationCostsSymmetric)) + "_allBestSettings" 
    matlabDict = {}
    matlabDict["allBestSettings"] = numpy.asmatrix(allBestSettings, dtype = numpy.int)
    scipy.io.savemat(outputFilename, matlabDict)
    
    
    if os.path.isfile(experimentSetting.MATLAB_DATA_FOLDER_RESULTS + "greedyMiser/" + dataName + "_" + str(int(misclassificationCostsSymmetric)) + "_forFinalTrainingAndTesting_" + str(4) + "_allResults.mat"):
    
        print("*************************** AFTER FINAL TRAINING *******************************************")
        
        testTotalCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
        testFeatureCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
        testMisClassificationCostsAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
        testAccuracyAllFolds = numpy.zeros(NUMBER_OF_FOLDS)
        
        for testFoldId in range(NUMBER_OF_FOLDS):
         
            allResultsInMatlab = scipy.io.loadmat(experimentSetting.MATLAB_DATA_FOLDER_RESULTS + "greedyMiser/" + dataName + "_" + str(int(misclassificationCostsSymmetric)) + "_forFinalTrainingAndTesting_" + str(testFoldId) + "_allResults")
            allAccTest = allResultsInMatlab['allAccTest']
            allTotalCost = allResultsInMatlab['allTotalCost'].transpose()
             
            bestTreeId = allBestSettings[testFoldId, 1]
            
            testAccuracy = allAccTest[0,bestTreeId]
            testFeatureCosts = allTotalCost[0,bestTreeId]
            totalCostsTestResult = evaluation.getTotalCostsSimple(testAccuracy, testFeatureCosts, misclassificationCostsSymmetric, sameClassCost)
        
            testTotalCostsAllFolds[testFoldId] = totalCostsTestResult
            testFeatureCostsAllFolds[testFoldId] = testFeatureCosts
            testMisClassificationCostsAllFolds[testFoldId] = (1.0 - testAccuracy) * misclassificationCostsSymmetric + testAccuracy * sameClassCost
            testAccuracyAllFolds[testFoldId] = testAccuracy
    
        print("*************************** AVERAGE OVER ALL FOLDS (misclassification costs = " + str(misclassificationCostsSymmetric) + ", sameClassCost = " + str(sameClassCost) + ") *******************************************")
        evaluation.showHelper("total costs = ", testTotalCostsAllFolds)
        evaluation.showHelper("feature costs = ", testFeatureCostsAllFolds)
        evaluation.showHelper("misclassification costs = ", testMisClassificationCostsAllFolds)
        evaluation.showHelper("accuracy = ", testAccuracyAllFolds)

        resultsRecorder.addAll(misclassificationCostsSymmetric, (testTotalCostsAllFolds, testFeatureCostsAllFolds, testMisClassificationCostsAllFolds, testAccuracyAllFolds))
        readyForSaving = True
        
if readyForSaving:
    resultsRecorder.writeOutResults(dataName + "_analyzeForGreedyMiser")

print("Finished Successfully")
import numpy
import matplotlib.pyplot as plt
import experimentHelper
import constants
import realdata
import evaluation

# prevents type 3 fonts
import matplotlib
import commonVisualizationDefinitions
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# *************************************************************************************************************************************
# ************* specify dataset and experimental setting here ****************************************
# *************************************************************************************************************************************
# 
dataName = "pima_5foldCV"
# dataName = "breastcancer_5foldCV"
# dataName = "pyhsioNetWithMissing_5foldCV"
# dataName = "heartDiseaseWithMissing_5foldCV"

FEATURE_SELECTION_METHOD = "nonLinearL1"
# FEATURE_SELECTION_METHOD = "greedy"

# COST_TYPE = "asymmetricCost"
# COST_TYPE = "symmetricCost"
COST_TYPE = "recall"
targetRecall = 0.95

# *************************************************************************************************************************************
# *************************************************************************************************************************************
# *************************************************************************************************************************************


classificationModelName = "Combined"
    
allMethodNames = commonVisualizationDefinitions.getAllMethodNames(dataName, FEATURE_SELECTION_METHOD, classificationModelName, COST_TYPE, AdaptGBRT_classifier = "Combined")

COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']

if COST_TYPE == "symmetricCost":
    print(dataName)
    print("FEATURE_SELECTION_METHOD = " + FEATURE_SELECTION_METHOD)
    for methodNameId in range(len(allMethodNames)):
        allTotalCosts, allFeatureCosts, allMisClassificationCosts, allAccuracies, allAUC, allRecall, allFDR, allOperationCosts, allRecall_atExactRecall, allFDR_atExactRecall, allOperationCosts_atExactRecall = experimentHelper.ResultsRecorder.readResults(dataName + "_" + allMethodNames[methodNameId] + "_" + COST_TYPE)
        assert(allTotalCosts.shape[0] == 2)
        rowIdFor400 = 0
        rowIdFor800 = 1
        result400Str = str(round(allTotalCosts[rowIdFor400,1],2)) + " (" + str(round(allTotalCosts[rowIdFor400,2],2)) +  ")"
        result800Str = str(round(allTotalCosts[rowIdFor800,1],2)) + " (" + str(round(allTotalCosts[rowIdFor800,2],2)) +  ")"
        
        print("\\bf " + commonVisualizationDefinitions.mapMethodToLabel(allMethodNames[methodNameId]) + " & " + result400Str + " & " + result800Str + " \\\\")
        # print("*****************")
        # print("method = ", allMethodNames[methodNameId])
        # print(str(allOperationCosts[rowIdFor400,0]) + " \t"  + result400Str) 
        # print(str(allOperationCosts[rowIdFor800,0]) + " \t"  + result800Str)

        
elif COST_TYPE == "asymmetricCost":
    
    plt.figure(figsize=(10, 15))
    
    axes_top = []
    for rowId in range(len(constants.allFalsePositiveCosts)):
        axes_top.append(plt.subplot2grid((3, len(constants.allFalsePositiveCosts)), (0, rowId), colspan=1))
    
    ax_middle = plt.subplot2grid((3, len(constants.allFalsePositiveCosts)), (1, 0), colspan=len(constants.allFalsePositiveCosts))
    ax_down = plt.subplot2grid((3, len(constants.allFalsePositiveCosts)), (2, 0), colspan=len(constants.allFalsePositiveCosts))
    
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.0, hspace = 0.5)
                                      
    barWidth = 25.0
    transparency = 0.9
    xMiddlePosition = 100
    
    trueLabelRatio = realdata.getAvgTrueLabelRatioOnTestData(dataName)
    print("trueLabelRatio = ", trueLabelRatio)
    x_ids = numpy.arange(len(constants.allFalsePositiveCosts))
    
    allMethodHandles = []
        
    print(dataName)
    print("FEATURE_SELECTION_METHOD = " + FEATURE_SELECTION_METHOD)
    for methodNameId in range(len(allMethodNames)):
        allTotalCosts, allFeatureCosts, allMisClassificationCosts, allAccuracies, allAUC, allRecall, allFDR, allOperationCosts, allRecall_atExactRecall, allFDR_atExactRecall, allOperationCosts_atExactRecall = experimentHelper.ResultsRecorder.readResults(dataName + "_" + allMethodNames[methodNameId] + "_" + COST_TYPE)
        print("*****************")
        print("method = ", allMethodNames[methodNameId])
        
        # color = [methodNameId]
        allWeigthedAccs = evaluation.getAllWeightedAccuracyies(trueLabelRatio, allMisClassificationCosts[:,1])
        methodHandle = ax_middle.errorbar(x = x_ids, y = allWeigthedAccs, linestyle=commonVisualizationDefinitions.ALL_LINESTYLES[methodNameId], color = COLOR_CYCLE[methodNameId], label = commonVisualizationDefinitions.mapMethodToLabel(allMethodNames[methodNameId]), marker = commonVisualizationDefinitions.ALL_MARKERS[methodNameId])
        ax_middle.xaxis.set_ticks(x_ids) 
        ax_middle.xaxis.set_ticklabels(numpy.asarray(constants.allFalsePositiveCosts, dtype = numpy.int)) 
        allMethodHandles.append(methodHandle)
        
        if not (dataName == "heartDiseaseWithMissing_5foldCV" and allMethodNames[methodNameId].startswith("getOptimalSequence_fullModel")):
            ax_down.errorbar(x = x_ids, y = allFeatureCosts[:,1], linestyle=commonVisualizationDefinitions.ALL_LINESTYLES[methodNameId], color = COLOR_CYCLE[methodNameId], label = commonVisualizationDefinitions.mapMethodToLabel(allMethodNames[methodNameId]), marker = commonVisualizationDefinitions.ALL_MARKERS[methodNameId])
            ax_down.xaxis.set_ticks(x_ids) 
            ax_down.xaxis.set_ticklabels(numpy.asarray(constants.allFalsePositiveCosts, dtype = numpy.int))
        
        
        # print("allWeigthedAccs = ", allWeigthedAccs)
        # assert(False)
        print("SET COSTS \t TOTAL COSTS \t OPERATION COSTS \t FDR \t RECALL \t FEATURE COSTS")
        
        assert(len(constants.allFalsePositiveCosts) == allOperationCosts.shape[0])
                
        for rowId in range(len(constants.allFalsePositiveCosts)):
            print("[" + str(allOperationCosts[rowId,0]) + "] \t"  + str(round(allTotalCosts[rowId,1],2)) + " (" + str(round(allTotalCosts[rowId,2],2)) +  ")  \t"  +  str(round(allOperationCosts[rowId,1],2)) + " (" + str(round(allOperationCosts[rowId,2],2)) +  ")"  + "\t" + str(round(allFDR[rowId,1],2)) + " (" + str(round(allFDR[rowId,2],2)) + ")" +  "\t" + str(round(allRecall[rowId,1],2)) + " (" + str(round(allRecall[rowId,2],2)) + ")" + "\t" + str(round(allFeatureCosts[rowId,1],2)) + " (" + str(round(allFeatureCosts[rowId,2],2)) +  ")")

            axes_top[rowId].bar([xMiddlePosition + methodNameId * barWidth], [allTotalCosts[rowId,1]], yerr = [allTotalCosts[rowId,2]], width=barWidth, color = COLOR_CYCLE[methodNameId], label=commonVisualizationDefinitions.mapMethodToLabel(allMethodNames[methodNameId]), alpha=transparency, ecolor='black', capsize=2)
            axes_top[rowId].set_xticks([xMiddlePosition + (len(allMethodNames) * barWidth / 2.0) - (barWidth / 2.0)])
            axes_top[rowId].set_xticklabels([str(int(constants.allFalsePositiveCosts[rowId]))])
            axes_top[rowId].spines['top'].set_visible(False)
            axes_top[rowId].spines['right'].set_visible(False)
        
            if rowId == 0:
                axes_top[rowId].set_ylabel('total costs', fontsize=10)
             
            if rowId == 3:
                axes_top[rowId].set_xlabel('false positive costs', fontsize=10)
              
            if dataName == "pima_5foldCV":
                if rowId == 0:
                    axes_top[rowId].set_ylim(0,10)
                elif rowId == 1:
                    axes_top[rowId].set_ylim(0,20)
                elif rowId== 2:
                    axes_top[rowId].set_ylim(0,30)
                elif rowId == 3:
                    axes_top[rowId].set_ylim(0,120)
                elif rowId == 4:
                    axes_top[rowId].set_ylim(0,200)
#                 elif rowId == 5:
#                     axes_top[rowId].set_ylim(0,150)
#                 elif rowId == 6:
#                     axes_top[rowId].set_ylim(0,300)
            elif dataName == "breastcancer_5foldCV":
                if rowId == 0:
                    axes_top[rowId].set_ylim(0,6)
                elif rowId == 1:
                    axes_top[rowId].set_ylim(0,8)
                elif rowId== 2:
                    axes_top[rowId].set_ylim(0,8)
                elif rowId == 3:
                    axes_top[rowId].set_ylim(0,15)
                elif rowId == 4:
                    axes_top[rowId].set_ylim(0,30)
                elif rowId == 5:
                    axes_top[rowId].set_ylim(0,150)
                elif rowId == 6:
                    axes_top[rowId].set_ylim(0,300)
                
            elif dataName == "pyhsioNetWithMissing_5foldCV":
                if rowId == 0:
                    axes_top[rowId].set_ylim(0,10)
                elif rowId == 1:
                    axes_top[rowId].set_ylim(0,20)
                elif rowId== 2:
                    axes_top[rowId].set_ylim(0,30)
                elif rowId == 3:
                    axes_top[rowId].set_ylim(0,100)
                elif rowId == 4:
                    axes_top[rowId].set_ylim(0,200)
            elif dataName == "heartDiseaseWithMissing_5foldCV":
                if rowId == 0:
                    axes_top[rowId].set_ylim(0,10)
                elif rowId == 1:
                    axes_top[rowId].set_ylim(0,20)
                elif rowId== 2:
                    axes_top[rowId].set_ylim(0,30)
                elif rowId == 3:
                    axes_top[rowId].set_ylim(0,100)
                elif rowId == 4:
                    axes_top[rowId].set_ylim(0,200)
#                 elif rowId == 5:
#                     axes_top[rowId].set_ylim(0,100)
#                 elif rowId == 6:
#                     axes_top[rowId].set_ylim(0,200)  
    
    print("---------------------------------")
    
    allFalsePositiveCosts_asString = [str(cost) for cost in constants.allFalsePositiveCosts]
    print("")
    print("dataName = ", dataName)
       
    
    ax_middle.set_xlabel('false positive costs', fontsize=10)
    ax_middle.set_ylabel('weigthed accuracy', fontsize=10)
    ax_middle.spines['top'].set_visible(False)
    ax_middle.spines['right'].set_visible(False)
    
    ax_down.set_xlabel('false positive costs', fontsize=10)
    ax_down.set_ylabel('# acquired covariates (costs)', fontsize=10)
    ax_down.spines['top'].set_visible(False)
    ax_down.spines['right'].set_visible(False)
    
    # plt.suptitle(constants.mapDataToLabel(dataName) + ", " + "false negative cost = " + str(int(constants.FN_TO_FP_RATIO)) + r" $\cdot$ " + "false positive costs")
    plt.suptitle(commonVisualizationDefinitions.mapDataToLabel(dataName) + ", " + "assymetric cost setting")
    plt.legend(handles=allMethodHandles)
    plt.show()
    
    # outputFilename = commonVisualizationDefinitions.OUTPUT_DIR + "result_" + dataName + "_" + COST_TYPE  + ".pdf"
    # plt.savefig(outputFilename)
    # print("wrote out file to " + outputFilename)

elif COST_TYPE == "recall":
    
    allTopValues = []
    allTopStds = []
    allxValues = []
    allxStds = []
    allyValues = []
    allyStds = []
        
    print(dataName)
    print("FEATURE_SELECTION_METHOD = " + FEATURE_SELECTION_METHOD)
    print("targetRecall = ", targetRecall)
    for methodNameId in range(len(allMethodNames)):
        allTotalCosts, allFeatureCosts, allMisClassificationCosts, allAccuracies, allAUC, allRecall, allFDR, allOperationCosts, allRecall_atExactRecall, allFDR_atExactRecall, allOperationCosts_atExactRecall = experimentHelper.ResultsRecorder.readResults(dataName + "_" + allMethodNames[methodNameId] + "_" + str(targetRecall) + COST_TYPE)
        print("*****************")
        print("method = ", allMethodNames[methodNameId])
        # print("allTotalCosts = ", allTotalCosts)
        # print("allAccuracies = ", allAccuracies)
        
        print("SET COSTS \t TOTAL COSTS \t OPERATION COSTS \t FDR \t RECALL \t FEATURE COSTS")
        
        topValues = numpy.zeros(len(constants.allFalsePositiveCosts))
        topStds = numpy.zeros(len(constants.allFalsePositiveCosts))
        
        xValues = numpy.zeros(len(constants.allFalsePositiveCosts))
        xStds = numpy.zeros(len(constants.allFalsePositiveCosts))
        yValues = numpy.zeros(len(constants.allFalsePositiveCosts))
        yStds = numpy.zeros(len(constants.allFalsePositiveCosts))
        
        assert(len(constants.allFalsePositiveCosts) == allOperationCosts.shape[0])
        
        for rowId in range(allOperationCosts.shape[0]):
            
            xValues[rowId] = allFeatureCosts[rowId,1]
            xStds[rowId] = allFeatureCosts[rowId,2]

            if allMethodNames[methodNameId].startswith("Shim2018") or allMethodNames[methodNameId].startswith("GreedyMiser"):
                # use results at exact recall (set to be the same as proposed method)
                print("[" + str(allOperationCosts[rowId,0]) + "] \t"  + str(round(allTotalCosts[rowId,1],2)) + " (" + str(round(allTotalCosts[rowId,2],2)) +  ")  \t"  +  str(round(allOperationCosts_atExactRecall[rowId,1],2)) + " (" + str(round(allOperationCosts_atExactRecall[rowId,2],2)) +  ")"  + "\t" + str(round(allFDR_atExactRecall[rowId,1],2)) + " (" + str(round(allFDR_atExactRecall[rowId,2],2)) + ")" +  "\t" + str(round(allRecall_atExactRecall[rowId,1],2)) + " (" + str(round(allRecall_atExactRecall[rowId,2],2)) + ")" + "\t" + str(round(allFeatureCosts[rowId,1],2)) + " (" + str(round(allFeatureCosts[rowId,2],2)) +  ")")
                
                topValues[rowId] = allOperationCosts_atExactRecall[rowId,1]
                topStds[rowId] = allOperationCosts_atExactRecall[rowId,2]
                
                yValues[rowId] = allFDR_atExactRecall[rowId,1]
                yStds[rowId] = allFDR_atExactRecall[rowId,2]

            else:
                # use actually observed recall
                print("[" + str(allOperationCosts[rowId,0]) + "] \t"  + str(round(allTotalCosts[rowId,1],2)) + " (" + str(round(allTotalCosts[rowId,2],2)) +  ")  \t"  +  str(round(allOperationCosts[rowId,1],2)) + " (" + str(round(allOperationCosts[rowId,2],2)) +  ")"  + "\t" + str(round(allFDR[rowId,1],2)) + " (" + str(round(allFDR[rowId,2],2)) + ")" +  "\t" + str(round(allRecall[rowId,1],2)) + " (" + str(round(allRecall[rowId,2],2)) + ")" + "\t" + str(round(allFeatureCosts[rowId,1],2)) + " (" + str(round(allFeatureCosts[rowId,2],2)) +  ")")

                topValues[rowId] = allOperationCosts[rowId,1]
                topStds[rowId] = allOperationCosts[rowId,2]
            
                yValues[rowId] = allFDR[rowId,1]
                yStds[rowId] = allFDR[rowId,2]
        
        allTopValues.append(topValues)
        allTopStds.append(topStds)
        allxValues.append(xValues)
        allxStds.append(xStds)
        allyValues.append(yValues)
        allyStds.append(yStds)
    
    print("---------------------------------")
    
    allFalsePositiveCosts_asString = [str(cost) for cost in constants.allFalsePositiveCosts]
    print("")
    print("dataName = ", dataName)
        
    plt.figure(figsize=(10, 15))
    
    axes_top = []
    for rowId in range(len(constants.allFalsePositiveCosts)):
        axes_top.append(plt.subplot2grid((3, len(constants.allFalsePositiveCosts)), (0, rowId), colspan=1))
    
    ax_middle = plt.subplot2grid((3, len(constants.allFalsePositiveCosts)), (1, 0), colspan=len(constants.allFalsePositiveCosts))
    ax_down = plt.subplot2grid((3, len(constants.allFalsePositiveCosts)), (2, 0), colspan=len(constants.allFalsePositiveCosts))
    
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.0, hspace = 0.5)
                                      
    barWidth = 25.0
    transparency = 0.9
    xMiddlePosition = 100
    
    methodNameId = 0  # assume that the first method is the proposed method !
    allTotalCosts, allFeatureCosts, allMisClassificationCosts, allAccuracies, allAUC, allRecall, allFDR, allOperationCosts, allRecall_atExactRecall, allFDR_atExactRecall, allOperationCosts_atExactRecall = experimentHelper.ResultsRecorder.readResults(dataName + "_" + allMethodNames[methodNameId] + "_" + str(targetRecall) + COST_TYPE)
    x_ids = numpy.arange(len(constants.allFalsePositiveCosts))
    ax_middle.errorbar(x = x_ids, y = allRecall[:,1], yerr = allRecall[:,2], color = COLOR_CYCLE[methodNameId], linestyle=commonVisualizationDefinitions.ALL_LINESTYLES[methodNameId], label = commonVisualizationDefinitions.mapMethodToLabel(allMethodNames[methodNameId]), marker = commonVisualizationDefinitions.ALL_MARKERS[methodNameId])
    ax_middle.xaxis.set_ticks(x_ids) 
    ax_middle.xaxis.set_ticklabels(numpy.asarray(constants.allFalsePositiveCosts, dtype = numpy.int)) 
    ax_middle.set_ylim([0.93, 1.001])
    ax_middle.yaxis.set_ticks([0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
    ax_middle.yaxis.set_ticklabels([0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
    ax_middle.axhline(y=0.95, color='k', linestyle='--')
    
    allMethodHandles = []
    
    for methodNameId in range(len(allMethodNames)):
        
        topValues = allTopValues[methodNameId]
        topStds = allTopStds[methodNameId]
        
        for rowId in range(len(constants.allFalsePositiveCosts)):
            methodHandleBar = axes_top[rowId].bar([xMiddlePosition + methodNameId * barWidth], [topValues[rowId]], yerr = [topStds[rowId]], width=barWidth, color = COLOR_CYCLE[methodNameId], label=commonVisualizationDefinitions.mapMethodToLabel(allMethodNames[methodNameId]), alpha=transparency, ecolor='black', capsize=2)
            axes_top[rowId].set_xticks([xMiddlePosition + (len(allMethodNames) * barWidth / 2.0) - (barWidth / 2.0)])
            axes_top[rowId].set_xticklabels([str(int(constants.allFalsePositiveCosts[rowId]))])
            axes_top[rowId].spines['top'].set_visible(False)
            axes_top[rowId].spines['right'].set_visible(False)
        
            if rowId == 0:
                axes_top[rowId].set_ylabel('operation costs', fontsize=10)
             
            if rowId == 3:
                axes_top[rowId].set_xlabel('false positive costs', fontsize=10)
            
            
            if dataName == "pima_5foldCV":
                if rowId == 0:
                    axes_top[rowId].set_ylim(0,6)
                elif rowId == 1:
                    axes_top[rowId].set_ylim(0,10)
                elif rowId== 2:
                    axes_top[rowId].set_ylim(0,15)
                elif rowId == 3:
                    axes_top[rowId].set_ylim(0,60)
                elif rowId == 4:
                    axes_top[rowId].set_ylim(0,70)
#                 elif rowId == 5:
#                     axes_top[rowId].set_ylim(0,150)
#                 elif rowId == 6:
#                     axes_top[rowId].set_ylim(0,300)
            elif dataName == "breastcancer_5foldCV":
                if rowId== 0:
                    axes_top[rowId].set_ylim(0,5)
                elif rowId == 1:
                    axes_top[rowId].set_ylim(0,10)
                elif rowId == 2:
                    axes_top[rowId].set_ylim(0,10)
                elif rowId == 3:
                    axes_top[rowId].set_ylim(0,20)
                elif rowId == 4:
                    axes_top[rowId].set_ylim(0,40)
                elif rowId == 5:
                    axes_top[rowId].set_ylim(0,100)
                elif rowId == 6:
                    axes_top[rowId].set_ylim(0,200)
                
            elif dataName == "pyhsioNetWithMissing_5foldCV":
                if rowId == 0:
                    axes_top[rowId].set_ylim(0,10)
                elif rowId == 1:
                    axes_top[rowId].set_ylim(0,20)
                elif rowId== 2:
                    axes_top[rowId].set_ylim(0,30)
                elif rowId == 3:
                    axes_top[rowId].set_ylim(0,50)
                elif rowId == 4:
                    axes_top[rowId].set_ylim(0,100)
            elif dataName == "heartDiseaseWithMissing_5foldCV":
                if rowId == 0:
                    axes_top[rowId].set_ylim(0,5)
                elif rowId == 1:
                    axes_top[rowId].set_ylim(0,10)
                elif rowId== 2:
                    axes_top[rowId].set_ylim(0,15)
                elif rowId == 3:
                    axes_top[rowId].set_ylim(0,40)
                elif rowId == 4:
                    axes_top[rowId].set_ylim(0,70)
                elif rowId == 5:
                    axes_top[rowId].set_ylim(0,300)
                elif rowId == 6:
                    axes_top[rowId].set_ylim(0,600)  


        if not (dataName == "heartDiseaseWithMissing_5foldCV" and allMethodNames[methodNameId].startswith("getOptimalSequence_fullModel")):
            methodHandleGraph = ax_down.errorbar(x = allxValues[methodNameId], y = allyValues[methodNameId], xerr=allxStds[methodNameId], yerr=allyStds[methodNameId], color = COLOR_CYCLE[methodNameId], linestyle=commonVisualizationDefinitions.ALL_LINESTYLES[methodNameId], label = commonVisualizationDefinitions.mapMethodToLabel(allMethodNames[methodNameId]), marker = commonVisualizationDefinitions.ALL_MARKERS[methodNameId])
        
        if not (dataName == "heartDiseaseWithMissing_5foldCV"):
            allMethodHandles.append(methodHandleGraph)
        else:
            allMethodHandles.append(methodHandleBar)
    
    
    ax_down.set_xlabel('# acquired covariates (costs)', fontsize=10)
    ax_down.set_ylabel('FDR', fontsize=10)
    ax_down.spines['top'].set_visible(False)
    ax_down.spines['right'].set_visible(False)
    
    ax_middle.set_xlabel('false positive costs', fontsize=10)
    ax_middle.set_ylabel('recall', fontsize=10)
    ax_middle.spines['top'].set_visible(False)
    ax_middle.spines['right'].set_visible(False)
    
    plt.suptitle(commonVisualizationDefinitions.mapDataToLabel(dataName) + ", " + r"recall $\geq$ " + str(targetRecall))
    plt.legend(handles=allMethodHandles)
    plt.show()
    
    # outputFilename = commonVisualizationDefinitions.OUTPUT_DIR + "result_" + dataName + "_" + COST_TYPE  +str(int(targetRecall * 100)) + ".pdf"
    # plt.savefig(outputFilename)
    # print("wrote out file to " + outputFilename)
    
else:

    assert(False)



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


def mapMethodToLabelForCMP(nameSpecifier):
    label = ""
    if nameSpecifier.startswith("getOptimalSequence_dynamic_BR_noUnlabeledData_nonLinearL1"):
        label = "AdaCOS (forward selection) "
    elif nameSpecifier.startswith("getOptimalSequence_dynamic_BR_noUnlabeledData_greedy"):
        label = "AdaCOS (group lasso) "
    else:
        return nameSpecifier
       
    return label

# dataName = "pima_5foldCV"
# dataName = "breastcancer_5foldCV"
# dataName = "pyhsioNetWithMissing_5foldCV"
# dataName = "heartDiseaseWithMissing_5foldCV"

allDataNames = ["pima_5foldCV", "breastcancer_5foldCV", "heartDiseaseWithMissing_5foldCV"]


classificationModelName = "Combined"
COST_TYPE = "asymmetricCost"




 
 
# allMethodNames = commonVisualizationDefinitions.getAllMethodNames(dataName, FEATURE_SELECTION_METHOD, classificationModelName, COST_TYPE)

allMethodNamesStemL1 = ["getOptimalSequence_dynamic_BR_noUnlabeledData_" + "nonLinearL1"] # , "getOptimalSequence_static_" + "nonLinearL1"]
allMethodNamesStemGreedy = ["getOptimalSequence_dynamic_BR_noUnlabeledData_" + "greedy"] # , "getOptimalSequence_static_" + "greedy"]

allMethodNamesStem = allMethodNamesStemL1 + allMethodNamesStemGreedy

allMethodNames = []
for methodName in allMethodNamesStem:
    if methodName.startswith("getOptimalSequence"):
        methodName += "_" + classificationModelName
    allMethodNames.append(methodName)

    
COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']

 
plt.figure(figsize=(10, 15))
# plt.tight_layout()

axes = []

for dataNameId in range(len(allDataNames)):
    axes.append([])
    for rowId in range(len(constants.allFalsePositiveCosts)):
        axes[dataNameId].append(plt.subplot2grid((3, len(constants.allFalsePositiveCosts)), (dataNameId, rowId), colspan=1))
    
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.0, hspace = 0.5)
                                 
barWidth = 25.0
transparency = 0.9
xMiddlePosition = 100


allMethodHandles = []

for dataNameId in range(len(allDataNames)):
    axes_top = axes[dataNameId]
    dataName = allDataNames[dataNameId]
    
    trueLabelRatio = realdata.getAvgTrueLabelRatioOnTestData(dataName)
    print("trueLabelRatio = ", trueLabelRatio)
    x_ids = numpy.arange(len(constants.allFalsePositiveCosts))

    for methodNameId in range(len(allMethodNames)):
        allTotalCosts, allFeatureCosts, allMisClassificationCosts, allAccuracies, allAUC, allRecall, allFDR, allOperationCosts, allRecall_atExactRecall, allFDR_atExactRecall, allOperationCosts_atExactRecall = experimentHelper.ResultsRecorder.readResults(dataName + "_" + allMethodNames[methodNameId] + "_" + COST_TYPE)
        print("*****************")
        print("method = ", allMethodNames[methodNameId])
        
        print("SET COSTS \t TOTAL COSTS \t OPERATION COSTS \t FDR \t RECALL \t FEATURE COSTS")
        
        assert(len(constants.allFalsePositiveCosts) == allOperationCosts.shape[0])
                
        for rowId in range(len(constants.allFalsePositiveCosts)):
            print("[" + str(allOperationCosts[rowId,0]) + "] \t"  + str(round(allTotalCosts[rowId,1],2)) + " (" + str(round(allTotalCosts[rowId,2],2)) +  ")  \t"  +  str(round(allOperationCosts[rowId,1],2)) + " (" + str(round(allOperationCosts[rowId,2],2)) +  ")"  + "\t" + str(round(allFDR[rowId,1],2)) + " (" + str(round(allFDR[rowId,2],2)) + ")" +  "\t" + str(round(allRecall[rowId,1],2)) + " (" + str(round(allRecall[rowId,2],2)) + ")" + "\t" + str(round(allFeatureCosts[rowId,1],2)) + " (" + str(round(allFeatureCosts[rowId,2],2)) +  ")")
    
            methodHandle = axes_top[rowId].bar([xMiddlePosition + methodNameId * barWidth], [allTotalCosts[rowId,1]], yerr = [allTotalCosts[rowId,2]], width=barWidth, color = COLOR_CYCLE[methodNameId * 7], label=mapMethodToLabelForCMP(allMethodNames[methodNameId]), alpha=transparency, ecolor='black', capsize=2)
            axes_top[rowId].set_xticks([xMiddlePosition + (len(allMethodNames) * barWidth / 2.0) - (barWidth / 2.0)])
            axes_top[rowId].set_xticklabels([str(int(constants.allFalsePositiveCosts[rowId]))])
            axes_top[rowId].spines['top'].set_visible(False)
            axes_top[rowId].spines['right'].set_visible(False)
        
            if rowId == 0:
                axes_top[rowId].set_ylabel('total costs', fontsize=10)
             
            if rowId == 3:
                axes_top[rowId].set_xlabel('false positive costs', fontsize=10)
        
            # axes_top[rowId].tight_layout()
            
        if dataNameId == 2:
            allMethodHandles.append(methodHandle)
            axes_top[3].legend(handles=allMethodHandles, loc='lower center', bbox_to_anchor=(6, -0.3))
    
        axes_top[3].set_title(commonVisualizationDefinitions.mapDataToLabel(dataName))
        
        
print("---------------------------------")

 

# plt.suptitle(constants.mapDataToLabel(dataName) + ", " + "false negative cost = " + str(int(constants.FN_TO_FP_RATIO)) + r" $\cdot$ " + "false positive costs")
# plt.tight_layout()

# plt.show()
outputFilename = commonVisualizationDefinitions.OUTPUT_DIR + "result_cmp_l1_and_greedy" + ".pdf"
plt.savefig(outputFilename)
print("wrote out file to " + outputFilename)
    



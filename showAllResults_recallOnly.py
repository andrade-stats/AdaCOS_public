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


allDataNames = ["pima_5foldCV", "breastcancer_5foldCV", "pyhsioNetWithMissing_5foldCV", "heartDiseaseWithMissing_5foldCV"]


classificationModelName = "Combined"
FEATURE_SELECTION_METHOD = "nonLinearL1"

COST_TYPE = "recall"
targetRecall = 0.95

allMethodNames = commonVisualizationDefinitions.getAllMethodNames(None, FEATURE_SELECTION_METHOD, classificationModelName, COST_TYPE, AdaptGBRT_classifier = "Combined")


COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']

 
plt.figure(figsize=(10, 15))
# plt.tight_layout()

axes = []

for dataNameId in range(len(allDataNames)):
    # axes.append([])
    # for rowId in range(len(constants.allFalsePositiveCosts)):
    axes.append(plt.subplot2grid((len(allDataNames), 1), (dataNameId, 0)))


# ax_middle = plt.subplot2grid((3, len(constants.allFalsePositiveCosts)), (1, 0), colspan=len(constants.allFalsePositiveCosts))
# ax_down = plt.subplot2grid((3, len(constants.allFalsePositiveCosts)), (2, 0), colspan=len(constants.allFalsePositiveCosts))
        
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.0, hspace = 0.5)

barWidth = 25.0
transparency = 0.9
xMiddlePosition = 100


allMethodHandles = []

for dataNameId in range(len(allDataNames)):
    ax_middle = axes[dataNameId]
    dataName = allDataNames[dataNameId]
    
    trueLabelRatio = realdata.getAvgTrueLabelRatioOnTestData(dataName)
    print("trueLabelRatio = ", trueLabelRatio)
    x_ids = numpy.arange(len(constants.allFalsePositiveCosts))

    allMethodHandles = []
    for methodNameId in range(len(allMethodNames)):
        print("*****************")
        print("method = ", allMethodNames[methodNameId])
        
        allTotalCosts, allFeatureCosts, allMisClassificationCosts, allAccuracies, allAUC, allRecall, allFDR, allOperationCosts, allRecall_atExactRecall, allFDR_atExactRecall, allOperationCosts_atExactRecall = experimentHelper.ResultsRecorder.readResults(dataName + "_" + allMethodNames[methodNameId] + "_" + str(targetRecall) + COST_TYPE)
        methodHandle = ax_middle.errorbar(x = x_ids, y = allRecall[:,1], yerr = allRecall[:,2], color = COLOR_CYCLE[methodNameId], linestyle=commonVisualizationDefinitions.ALL_LINESTYLES[methodNameId], label = commonVisualizationDefinitions.mapMethodToLabel(allMethodNames[methodNameId]), marker = commonVisualizationDefinitions.ALL_MARKERS[methodNameId])
        
        print("SET COSTS \t TOTAL COSTS \t OPERATION COSTS \t FDR \t RECALL \t FEATURE COSTS")
        
        assert(len(constants.allFalsePositiveCosts) == allOperationCosts.shape[0])
        allMethodHandles.append(methodHandle)
            
    ax_middle.xaxis.set_ticks(x_ids) 
    ax_middle.xaxis.set_ticklabels(numpy.asarray(constants.allFalsePositiveCosts, dtype = numpy.int))
    
    START = 0.81 # 0.93
    STOP = 1.001
    STEP = 0.02
    steps_for_ticks = numpy.arange(start = START, stop=STOP, step = 0.01)
    # steps_for_label = numpy.arange(start = START, stop=STOP, step = STEP)
    
    print("steps_for_ticks = ", steps_for_ticks)
    # steps = steps.tolist()
    # steps.append(1.0)
    
    steps_label = []
    for i, s in enumerate(steps_for_ticks):
        if i % 2 == 0:
            steps_label.append(str(round(s,2)))
        else:
            steps_label.append("")
    # print("steps_label = ", steps_label)
    # assert(False)
    # steps = [0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
    # print("steps = ", steps)
    # assert(False)
     
    ax_middle.set_ylim([START, STOP])
    ax_middle.yaxis.set_ticks(steps_for_ticks)
    ax_middle.yaxis.set_ticklabels(steps_label)
    ax_middle.axhline(y=targetRecall, color='k', linestyle='--') 
    
    
    ax_middle.set_ylabel('recall', fontsize=10)
    ax_middle.set_xlabel('false positive costs', fontsize=10)
    
    ax_middle.spines['top'].set_visible(False)
    ax_middle.spines['right'].set_visible(False)
    
    if dataNameId == 3:
        ax_middle.legend(handles=allMethodHandles, loc='lower center', bbox_to_anchor=(1, -0.8)) #  bbox_to_anchor=(6, -0.3))
     
    ax_middle.set_title(commonVisualizationDefinitions.mapDataToLabel(dataName))
        
        
print("---------------------------------")

 

# plt.suptitle(constants.mapDataToLabel(dataName) + ", " + "false negative cost = " + str(int(constants.FN_TO_FP_RATIO)) + r" $\cdot$ " + "false positive costs")
# plt.tight_layout()

# plt.show()
outputFilename = commonVisualizationDefinitions.OUTPUT_DIR + "result_onlyRecall" + str(int(targetRecall * 100)) + ".pdf"
plt.savefig(outputFilename)
print("wrote out file to " + outputFilename)

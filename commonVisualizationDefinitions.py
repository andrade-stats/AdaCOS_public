
ALL_LINESTYLES = ['-', '--', '-.', ':', ":", ":"]
ALL_MARKERS = ["^", "o", "s", ">", "P", "p"]

OUTPUT_DIR = "/export/home/s-andrade/newStart/eclipseWorkspaceDynamic/DynamicCovariateSelection/latex/"

def getAllMethodNames(dataName, FEATURE_SELECTION_METHOD, classificationModelName, COST_TYPE, AdaptGBRT_classifier = None):
    
    if COST_TYPE == "recall":
        allMethodNamesStem = ["getOptimalSequence_dynamic_BR_noUnlabeledData_" + FEATURE_SELECTION_METHOD, "getOptimalSequence_static_" + FEATURE_SELECTION_METHOD, "getOptimalSequence_fullModel", "Shim2018", "GreedyMiser"]
    elif COST_TYPE == "asymmetricCost" or COST_TYPE == "symmetricCost": 
        allMethodNamesStem = ["getOptimalSequence_dynamic_BR_noUnlabeledData_" + FEATURE_SELECTION_METHOD, "getOptimalSequence_static_" + FEATURE_SELECTION_METHOD, "getOptimalSequence_fullModel", "Shim2018", "GreedyMiser", "AdaptGBRT_" + AdaptGBRT_classifier]
    elif COST_TYPE == "cmp":
        assert(False)
        # allMethodNamesStem = ["getOptimalSequence_dynamic_BR_noUnlabeledData_" + FEATURE_SELECTION_METHOD1, "getOptimalSequence_static_" + FEATURE_SELECTION_METHOD]
    else:
        assert(False)
    
    
    allMethodNames = []
    for methodName in allMethodNamesStem:
        
        if methodName.startswith("getOptimalSequence"):
            methodName += "_" + classificationModelName
            
        allMethodNames.append(methodName)
    
    return allMethodNames
        
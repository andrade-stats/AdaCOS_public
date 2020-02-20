
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


def mapMethodToLabel(nameSpecifier):
    label = ""
    if nameSpecifier.startswith("getOptimalSequence_dynamic_BR_noUnlabeledData_"):
        label = "AdaCOS"
    elif nameSpecifier.startswith("getOptimalSequence_static_"):
        label = "COS"
    elif nameSpecifier.startswith("getOptimalSequence_fullModel"):
        label = "Full Model"
    elif nameSpecifier.startswith("AdaptGBRT_"):
        label = "AdaptGbrt"
    else:
        return nameSpecifier

        
    return label


def mapDataToLabel(nameSpecifier):
    if nameSpecifier == "breastcancer_5foldCV":
        return "Breast Cancer"
    elif nameSpecifier == "pima_5foldCV":
        return "Diabetes"
    elif nameSpecifier == "heartDiseaseWithMissing_5foldCV":
        return "Heart Disease"
    elif nameSpecifier == "pyhsioNetWithMissing_5foldCV":
        return "PhysioNet"
    else:
        assert(False)
        
        
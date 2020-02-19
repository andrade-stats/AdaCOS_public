
NUMBER_OF_FOLDS = 5
IMPUTATION_METHOD = "gaussian_imputation"

allFalsePositiveCosts = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

FN_TO_FP_RATIO = 10.0 # i.e. false negative costs = 10 * false positive costs

BASE_FOLDER = "/export/home/s-andrade/newStart/eclipseWorkspaceDynamic/preparedData/"


NUMBER_OF_TRAIN_EVAL_FOLDS_MATLAB_METHODS = 5 # used by Greedy miser and AdaptGBRT

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


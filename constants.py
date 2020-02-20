
# number of folds for the outer-loop cross-validation
NUMBER_OF_FOLDS = 5

# method used for imputation, when missing values are present
IMPUTATION_METHOD = "gaussian_imputation"

# user-specified false positive costs that are used for training and evaluation
allFalsePositiveCosts = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

# user-specified false negative costs = 10 * false positive costs
FN_TO_FP_RATIO = 10.0 


ABSOLUTE_PATH_PREFIX = "/export/home/s-andrade/newStart/eclipseWorkspaceDynamic/AdaCOS_public/"

# folder containing the original datasets
REAL_DATA_FOLDER = ABSOLUTE_PATH_PREFIX + "datasets/"

# folder containing all preprocessed data
BASE_FOLDER = ABSOLUTE_PATH_PREFIX + "preparedData/"

# folder used for saving the model of each classifier
MODEL_FOLDERNAME =  ABSOLUTE_PATH_PREFIX + "models/"

# folder where the evaluation results are written
STANDARD_RESULTS_FOLDER = ABSOLUTE_PATH_PREFIX + "eclipseWorkspaceDynamic/DynamicCovariateSelection/results/"

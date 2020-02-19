
allAttributeNames = ['Height', 'Bilirubin', 'MAP', 'ALT', 'Cholesterol', 'DiasABP', 'Mg', 'pH', 'TroponinI', 'BUN', 'Temp', 'Glucose', 'HCT', 'MechVent', 'K', 'PaO2', 'NIDiasABP', 'HCO3', 'Na', 'Lactate', 'ALP', 'Urine', 'PaCO2', 'Weight', 'Platelets', 'Albumin', 'RespRate', 'Creatinine', 'NISysABP', 'SysABP', 'WBC', 'SaO2', 'AST', 'NIMAP', 'ICUType', 'GCS', 'Gender', 'Parameter', 'HR', 'Age', 'TroponinT', 'FiO2']

covariateNameToIndex = {}
for i, name in enumerate(allAttributeNames):
    covariateNameToIndex[name] = i

filenameData = "physioNet_allData"
filenameLabels = "physioNet_allLabels"

VARIABLE_SUBSET_FOR_EVALUATION_NO_MISSING = set([34, 35, 36, 38, 39, 9, 10, 11, 12, 14, 17, 18, 21, 24, 27, 30])
VARIABLE_SUBSET_FOR_EVALUATION_WITH_MISSING = set([2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 27, 28, 29, 30, 33, 34, 35, 36, 38, 39, 41])
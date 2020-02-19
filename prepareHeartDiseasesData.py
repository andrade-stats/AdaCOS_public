import numpy
import constants

# uses the heart-disease data from 
# https://archive.ics.uci.edu/ml/datasets/heart+Disease
# and the costs as defined in 
# "Cost-Sensitive Classification: Empirical Evaluation of a Hybrid Genetic Decision Tree Induction Algorithm", Turney

RAW_DATA_FOLDER = "/export/home/s-andrade/eclipseWorkspaceDynamic/heart-diseases/"


def loadData(filename, NUMBER_OF_ATTRIBUTES):
    
    allDataRows = []
    allLabels = []
    
    with open(RAW_DATA_FOLDER + filename, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            allAttributesPlusLabel = line.split(",")
            assert(len(allAttributesPlusLabel) == NUMBER_OF_ATTRIBUTES + 1)
            label_5step = int(allAttributesPlusLabel[NUMBER_OF_ATTRIBUTES])
            assert(label_5step >= 0 and label_5step <= 4)
            if label_5step == 0:
                # no heart-disease
                label = 0
            else:
                # some heart-disease
                label = 1
            
            allLabels.append(label)
            
            dataRow = numpy.zeros(NUMBER_OF_ATTRIBUTES) * numpy.nan
            for attrId in range(NUMBER_OF_ATTRIBUTES):
                if allAttributesPlusLabel[attrId] != "?":
                    dataRow[attrId] = float(allAttributesPlusLabel[attrId])
                else:
                    print("CONTAINS ? IN ROW" + str(i))
            allDataRows.append(dataRow)
            
    
    TOTAL_NUMBER_OF_SAMPLES = len(allDataRows)
    assert(TOTAL_NUMBER_OF_SAMPLES == len(allLabels))
    
    data = numpy.zeros((TOTAL_NUMBER_OF_SAMPLES, NUMBER_OF_ATTRIBUTES)) * numpy.nan
    labels = numpy.zeros(TOTAL_NUMBER_OF_SAMPLES, dtype = numpy.int) * numpy.nan
    
    for i in range(TOTAL_NUMBER_OF_SAMPLES):
        data[i] = allDataRows[i]
        labels[i] = allLabels[i]
    
    return data, labels




NUMBER_OF_ATTRIBUTES = 13

# ALL_FILENAMES_FOR_UNLABELED_DATA = ["processed.hungarian.data", "processed.switzerland.data", "processed.va.data"]
filename = "processed.cleveland.data"

# for filename in ALL_FILENAMES_FOR_UNLABELED_DATA:
data, labels = loadData(filename, NUMBER_OF_ATTRIBUTES)    
                    


assert(not numpy.any(numpy.isnan(labels)))
numpy.save(constants.BASE_FOLDER + "heartDisease_ClevelandData", data)
numpy.save(constants.BASE_FOLDER + "heartDisease_ClevelandLabels", labels)

print("SAVED ALL SUCESSFULLY")

print("subset of data:")
print(data[0:10])
print("subset of labels:")
print(labels[0:10])



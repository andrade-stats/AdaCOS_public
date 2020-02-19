import os
import numpy
import physioNet

# prepares the data available at
# https://archive.physionet.org/pn3/challenge/2012/
# (former address: https://physionet.org/pn3/challenge/2012/ )

# set this to the folder where the raw data is:
RAW_DATA_FOLDER = "/export/home/s-andrade/eclipseWorkspaceDynamic/rawData"


# use the last recorded value as in "Joint Active Feature Acquisition and Classification with Variable-Size Set-Encoding", NIPS 2018
def loadPatientData(data, covariateNameToIndex, recordIdToSampleId, allRecordIds, dataSetMarker):
    
    currentSet = "/set-" + dataSetMarker + "/"

    # attributeObservationCount = numpy.zeros(len(covariateNameToIndex))
    
    for root, dirs, files in os.walk(RAW_DATA_FOLDER + currentSet):  
        for filename in files:
            assert(filename.endswith(".txt"))
            
            # recordedForThisPatient = set()
            
            with open(RAW_DATA_FOLDER + currentSet + filename, "r") as f:
                recordId = int(filename.split(".")[0])
                allRecordIds.add(recordId)
                for i, line in enumerate(f):
                    
                    if i == 0:
                        continue
                    
                    line = line.strip()
                    # format = 00:00,Weight,97
                    attributeName = line.split(",")[1]
                    attributeValue = line.split(",")[2]
     
                    if attributeName == "":
                        print("warning ignore this line:")
                        print(line)
                        continue
                   
                    if attributeName == "RecordID":
                        assert(recordId == int(attributeValue))
                        continue
                    
                    assert(attributeName in covariateNameToIndex)
                    assert(recordId in recordIdToSampleId)
                    
                    attributeValue = float(attributeValue)
                    
                    if attributeValue >= 0:
                        sampleId = recordIdToSampleId[recordId]
                        covariateId = covariateNameToIndex[attributeName]
                        data[sampleId, covariateId] = attributeValue
                    else:
                        # unknown value
                        if attributeValue != -1:
                            print("warning ignore this line:")
                            print(line)
                        else: 
                            assert(attributeValue == -1)
                        
                    # if attributeName not in recordedForThisPatient: 
                    #    recordedForThisPatient.add(attributeName)
                    #    attributeObservationCount[covariateNameToIndex[attributeName]] += 1
                    
    
        
    return data, allRecordIds





TOTAL_NUMBER_OF_SAMPLES = 12000 

data = numpy.zeros((TOTAL_NUMBER_OF_SAMPLES, len(physioNet.allAttributeNames))) * numpy.nan
labels = numpy.zeros(TOTAL_NUMBER_OF_SAMPLES, dtype = numpy.int) * numpy.nan


recordIdToSampleId = {}
patientCount = 0

# get all labels
for dataSetMarker in ["a", "b", "c"]:
    with open(RAW_DATA_FOLDER + "/Outcomes-" + dataSetMarker + ".txt", "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0:
                continue
            
            recordId = int(line.split(",")[0])
            deathFlag = int(line.split(",")[5])  # 0: survival, or 1: in-hospital death
            assert(recordId > 0)
            assert(deathFlag == 0 or deathFlag == 1)
            
            assert(recordId not in recordIdToSampleId)
            recordIdToSampleId[recordId] = patientCount
            labels[patientCount] = deathFlag
            patientCount += 1
        

print("number of labeled samples = ", len(recordIdToSampleId))


allRecordIds = set()
for dataSetMarker in ["a", "b", "c"]:
    data, allRecordIds = loadPatientData(data, physioNet.covariateNameToIndex, recordIdToSampleId, allRecordIds, dataSetMarker)


# print("number of all samples = ", patientCount)
# print("all record ids = ", len(allRecordIds))
# print("allAttributeNames = ")
# print(allAttributeNames)
# print("size = ")
# print(len(allAttributeNames))
# print("FINISHED SUCCESSFUL")

assert(not numpy.any(numpy.isnan(labels)))

numpy.save(physioNet.filenameData, data)
numpy.save(physioNet.filenameLabels, labels)

print("SAVED ALL SUCESSFULLY")

print("subset of data:")
print(data[0:10])
print("subset of labels:")
print(labels[0:10])



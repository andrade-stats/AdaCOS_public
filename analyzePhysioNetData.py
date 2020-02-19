import os
import numpy
import physioNet
import collections
import constants

def getAttributePosToOccurenceIds(sampleIdToPosIds):
    
    posToOccSet = collections.defaultdict(set)
    
    for sampleId, posIdSet in sampleIdToPosIds.items():
        # print(sampleId)
        # print(posIdSet)
        
        for pos in posIdSet:
            posToOccSet[pos].add(sampleId)
            
            
    return posToOccSet


def filterWithCommonSubset(sampleIdToPosIds, commonSubset):
    sampleIdToPosIdsFiltered = {}
    for sampleId, posIdSet in sampleIdToPosIds.items():
        if posIdSet.issuperset(commonSubset):
            sampleIdToPosIdsFiltered[sampleId] = posIdSet
    
    return sampleIdToPosIdsFiltered


def getMostFrequentNew(posToOccSet, commonSubset):
    sortedPosToOccSet = sorted(posToOccSet.items(), key=lambda x: -len(x[1]))
    for posId, occSet in sortedPosToOccSet:
        if posId not in commonSubset:
            print("number of samples is now = ", len(occSet))
            return posId
    
    return None


data = numpy.load(constants.BASE_FOLDER + physioNet.filenameData + ".npy")
labels = numpy.load(constants.BASE_FOLDER + physioNet.filenameLabels + ".npy")

print("data.shape = ", data.shape)

posRatio = numpy.sum(labels) / labels.shape[0]
print("posRatio = ", posRatio)


print("allAttributeNames = ", len(physioNet.allAttributeNames))

print("subset of data:")
print(data[0:10])
print("subset of labels:")
print(labels[0:10])


# assert(False)

sampleIdToPosIds = {}
for i in range(data.shape[0]):
    availablePositions = numpy.where(numpy.logical_not(numpy.isnan(data[i])))[0]
    # countPattern[numpy.str(availablePositions)] += 1
    sampleIdToPosIds[i] =  set(availablePositions)
# print(e)


# posId = 34, number of occurrences = 12000
# posId = 39, number of occurrences = 12000
# posId = 36, number of occurrences = 11988

print("FNISHED:")



commonSubset = set([34,39,36])

while(True):
    sampleIdToPosIds = filterWithCommonSubset(sampleIdToPosIds, commonSubset)
    posToOccSet = getAttributePosToOccurenceIds(sampleIdToPosIds)
    posId = getMostFrequentNew(posToOccSet, commonSubset)
    if posId is None:
        break
    commonSubset.add(posId)
    print("posId = ", posId)
    print("commonSubset size = ", len(commonSubset))
    # if posId == 19:
    #    print(commonSubset)

# sortedPosToOccSet = sorted(posToOccSet.items(), key=lambda x: -len(x[1]))
# print("Top elems: ")
# for posId, occSet in sortedPosToOccSet:
#     print("posId = " +  str(posId) + ", number of occurrences = " + str(len(occSet)))

VARIABLE_SUBSET_FOR_EVALUATION_NO_MISSING = set([34, 35, 36, 38, 39, 9, 10, 11, 12, 14, 17, 18, 21, 24, 27, 30])
print("")
print("VARIABLE_SUBSET_FOR_EVALUATION_NO_MISSING = ", VARIABLE_SUBSET_FOR_EVALUATION_NO_MISSING)
print("size = ", len(VARIABLE_SUBSET_FOR_EVALUATION_NO_MISSING))

VARIABLE_SUBSET_FOR_EVALUATION_WITH_MISSING = set([2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 27, 28, 29, 30, 33, 34, 35, 36, 38, 39, 41])
print("")
print("VARIABLE_SUBSET_FOR_EVALUATION_WITH_MISSING = ", VARIABLE_SUBSET_FOR_EVALUATION_WITH_MISSING)
print("size = ", len(VARIABLE_SUBSET_FOR_EVALUATION_WITH_MISSING))

# commonSubset size =  30
# number of samples is now =  2236


import numpy as np
import pandas as pd
import operator

def KNNclassify(inx,dataSet,labels,k):
	diffMat = np.tile(inx,(dataSet.shape[0],1)) - dataSet
	distances = ((diffMat**2).sum(axis = 1))**0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		#if voteIlabel not in classCount,set 0,else count plus 1
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse = True)
	return sortedClassCount[0][0]

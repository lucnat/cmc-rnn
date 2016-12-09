#
# A very simple class to load batches
# Written by Luca Naterop, Zuerich, 2016
#

import math
import numpy as np

class BatchLoader:
	'Creates batches from data and returns them. Can also create batches with corresponding targets for RNNs'
	def __init__(self, data, batchsize):
		'batchsize might shrink if we are at the end of the data'
		self.data = data
		self.batchsize = batchsize
		self.pointer = 0

	def nextBatch(self):
		'returns (nextBatch, isLast) where isLast is true if nextBatch is the last batch and otherwise false'
		if self.pointer + self.batchsize > len(self.data):
			# then it was the last batch, return all till end
			next = self.data[self.pointer:]
			self.pointer = 0
			return next, True
		else:
			#it was not the last batch, so everything is nice and normal
			next = self.data[self.pointer:(self.pointer+self.batchsize)]
			self.pointer += self.batchsize
			return next, False

	def nextRNNBatch(self):
		'returns (inputs, targets, isLast) where isLast ist true if that was the last one it could construct with corresponding targets'
		if self.pointer + self.batchsize + 1 >= len(self.data):
			# then it was the last batch, because targets reaches until the end of data, that is, includes last data point
			inputs = self.data[self.pointer : len(self.data)-1]
			targets = self.data[self.pointer+1 :] 
			self.pointer = 0
			return inputs, targets, True
		else:
			#it was not the last batch, so everything is nice and normal
			inputs = self.data[ self.pointer : (self.pointer+self.batchsize) ]
			targets = self.data[ (self.pointer+1) : (self.pointer+self.batchsize+1) ]			
			self.pointer += self.batchsize
			return inputs, targets, False


	def amountOfBatches(self):
		amount = float(len(self.data))/self.batchsize
		return int(math.ceil(amount))

	def resetPointer(self):
		self.pointer = 0



# Usage: Uncomment and see for yourself

# x = np.random.rand(10,10)
# print(tabulate(x))
# loader = BatchLoader(x,3)
# for i in range(3):
# 	inputs, targets, isTrue = loader.nextRNNBatch()
# 	print(" =============================================== ")
# 	print(tabulate(inputs))
# 	print(tabulate(targets))






import math, os, struct
from array import array as pyarray
from pylab import *
from numpy import *
from random import sample
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from scipy import signal
import numpy

# kron divide ke jagah, kron ==, remove unzip

# def increasePad(array):
# 	return numpy.lib.pad(array, (2,2), 'constant', constant_values=0)

def loadMNIST(dataset="training", digits=arange(10), path="."):
	# This function is taken from http://g.sweyla.com/blog/2012/mnist-numpy/
	if dataset == "training":
		fname_img = os.path.join(path, 'train-images.idx3-ubyte')
		fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
	elif dataset == "testing":
		fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
		fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
	else:
		raise ValueError("dataset must be 'testing' or 'training'")
	lbl_file = open(fname_lbl, 'rb')
	magic_nr, size = struct.unpack(">II", lbl_file.read(8))
	lbl = pyarray("b", lbl_file.read())
	lbl_file.close()
	img_file = open(fname_img, 'rb')
	magic_nr, size, rows, cols = struct.unpack(">IIII", img_file.read(16))
	img = pyarray("B", img_file.read())
	img_file.close()
	ind = [ k for k in range(size) if lbl[k] in digits ]
	N = len(ind)
	images = zeros((N, rows+4, cols+4), dtype=uint8)
	labels = zeros((N, 1), dtype=int8)
	for i in range(len(ind)):
		tmp = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape(rows,cols)
		images[i] = numpy.lib.pad(tmp, (2,2), 'constant', constant_values=0)
		labels[i] = lbl[ind[i]]
	return images, labels

def reluFunction(x):
	return max(0.1*x,x)

def reluDeriFunction(x):
	if x<=0:return 0.1
	else:return 1


class LENET5:

	def __init__(self, inputSize):
		self.relu = vectorize(reluFunction)
		self.reluDerivative = vectorize(reluDeriFunction)
		self.convWeight1 = random.rand(6,5,5) * 0.001
		self.conWeightGradient1 = zeros((6,5,5))
		self.convOut1 = zeros((6,28,28))
		self.reluOut1 = zeros((6,28,28))
		self.reluDer1 = zeros((6,28,28))
		self.poolOut1 = zeros((6,14,14))
		self.poolMaxIndex1 = zeros((6,14,14))
		self.convWeight2 = random.rand(16,6, 5,5) * 0.001
		self.conWeightGradient2 = zeros((16, 6, 5, 5))
		self.convOut2 = zeros((16,10,10))
		self.reluOut2 = zeros((16,10,10))
		self.reluDer2 = zeros((16,10,10))
		self.poolOut2 = zeros((16,5,5))
		self.poolMaxIndex2 = zeros((16,5,5))
		self.fcWeight1 = random.rand(400,120) * 0.001
		self.fcWeightGradient1 =  zeros((400,120))
		self.fcOut1 = zeros(120)
		self.reluOut3 = zeros((120))
		self.reluDer3 = zeros((120))
		self.fcWeight2 = random.rand(120,84) * 0.001
		self.fcWeightGradient2 = zeros((120,84))
		self.fcOut2 = zeros((84))
		self.reluOut4 = zeros((84))
		self.reluDer4 = zeros((84))
		self.fcWeight3 = random.rand(84,10) * 0.001
		self.fcWeightGradient3 = zeros((84,10))
		self.fcOut3 = zeros((10))
		self.softMaxOut = zeros(10)

	def forwardConv(self, input, filters):
		output = zeros((filters.shape[0], input.shape[1]+1-filters.shape[2], input.shape[1]+1-filters.shape[2])) 
		for i in range(0, filters.shape[0]):
			if len(input.shape) == 3:
				for j in range(0, input.shape[0]):
					output[i] = output[i] +  signal.convolve2d(input[j], rot90(rot90(filters[i][j])), mode= 'valid')
			else:
				output[i] = signal.convolve2d(input, rot90(rot90(filters[i])), mode = 'valid')
		return output


	def weightGradConv(self, input, filters):
		if len(input.shape) == 3:
			output = zeros((filters.shape[0], input.shape[0], input.shape[1]-filters.shape[1]+1, input.shape[2]-filters.shape[2]+1))
			for i in range(0, filters.shape[0]):
				for j in range(0, input.shape[0]):
					output[i][j] = signal.convolve2d(input[j], rot90(rot90(filters[i])), mode = 'valid')
		else:
			output = zeros((filters.shape[0], input.shape[0]-filters.shape[1]+1, input.shape[1]-filters.shape[2]+1))
			for i in range(0, filters.shape[0]):
					output[i] = signal.convolve2d(input, rot90(rot90(filters[i])), mode = 'valid')
		return output

	def backpropConv(self, input, filters):
		output = zeros((filters.shape[1], input.shape[1] + filters.shape[2] - 1, input.shape[2] + filters.shape[3] - 1))
		for i in range(0,filters.shape[0]):
			for j in range(0,filters.shape[1]):
				output[j] += signal.convolve2d(input[i], filters[i][j], mode = 'full')
		return output

	def reluLayer(self, input):
		return self.relu(input)

	def poolLayer(self, input, receptiveField):
		return block_reduce(input, block_size=(1,receptiveField,receptiveField), func=numpy.max)
	
	def softmaxLayer(self, input): #done
		exp = vectorize(math.exp)
		tmp = exp(input)
		tmp /= numpy.sum(tmp)
		return tmp

	def forwardFeed(self, inputImg, lbl): #done 
		#print "Convweight1", self.convWeight1
		self.convOut1 = self.forwardConv(inputImg, self.convWeight1)
		self.reluOut1 = self.relu(self.convOut1)
		self.reluDer1 = self.reluDerivative(self.convOut1)
		self.poolOut1 = self.poolLayer(self.reluOut1, 2)
		self.poolMaxIndex1 = vectorize(int)(self.reluOut1 == kron(self.poolOut1, ones((1,2, 2))))
		self.convOut2 = self.forwardConv(self.poolOut1, self.convWeight2)
		self.reluOut2 = self.relu(self.convOut2)
		self.reluDer2 = self.reluDerivative(self.convOut2)
		self.poolOut2 = self.poolLayer(self.reluOut2, 2)
		self.poolMaxIndex2 = vectorize(int)(self.reluOut2 == kron(self.poolOut2, ones((1,2,2))))
		self.fcOut1 = dot(self.poolOut2.reshape(1,400), self.fcWeight1)
		self.reluOut3 = self.relu(self.fcOut1)
		self.reluDer3 = self.reluDerivative(self.fcOut1)
		self.fcOut2 = dot(self.reluOut3, self.fcWeight2)
		self.reluOut4 = self.relu(self.fcOut2)
		self.reluDer4 = self.reluDerivative(self.fcOut2)
		self.fcOut3 = dot(self.reluOut4, self.fcWeight3)
		self.softMaxOut = self.softmaxLayer(self.fcOut3.reshape(10))
		return -1*math.log(self.softMaxOut[lbl])		

	def backwardFeed(self, inputImg, lbl):
		self.softMaxOut[lbl] -= 1
		fcBackInput3 = self.softMaxOut #10
#		print "FC3", fcBackInput3
		self.fcWeightGradient3 += dot(self.reluOut4.reshape(84,1), fcBackInput3.reshape(1,10)) #84*10
		#print "FCWG3", self.fcWeightGradient3
		reluBackInput4 = dot(self.fcWeight3, fcBackInput3.reshape(10,1)) #84*1
#		print "R4", reluBackInput4
		fcBackInput2 = multiply(self.reluDer4.reshape(84,1), reluBackInput4) #84*1
#		print "FC2", fcBackInput2
		self.fcWeightGradient2 += dot(self.reluOut3.reshape(120,1), fcBackInput2.reshape(1,84)) #120*84
		#print "FCWG2", self.fcWeightGradient2
		reluBackInput3 = dot(self.fcWeight2, fcBackInput2.reshape(84,1)) #120*1
#		print "R3", reluBackInput3
		fcBackInput1 = multiply(self.reluDer3.reshape(120,1), reluBackInput3) #500*1
#		print "FC1", fcBackInput1
		self.fcWeightGradient1 += dot(self.poolOut2.reshape(400,1), fcBackInput1.reshape(1,120)) # 800*500
		#print "FCWG1", self.fcWeightGradient1
		poolBackInput2 = dot(self.fcWeight1, fcBackInput1).reshape(16,5,5) # 16*4*4
#		print "P2", poolBackInput2
		reluBackInput2 = multiply(kron(poolBackInput2, numpy.ones((1, 2,2))), self.poolMaxIndex2) #16*8*8
#		print "R2", reluBackInput2
		convBackInput2 = multiply(self.reluDer2, reluBackInput2) #16*8*8
#		print "C2", convBackInput2
		self.conWeightGradient2 += self.weightGradConv(self.poolOut1, convBackInput2) #16*6*5*5
		#print "CWG2", self.conWeightGradient2
		poolBackInput1 = self.backpropConv(self.convOut2, self.convWeight2) #6*12*12
#		print "P1", poolBackInput1
		reluBackInput1 = multiply(kron(poolBackInput1, numpy.ones((1, 2,2))), self.poolMaxIndex1) #6*24*24
#		print "R1", reluBackInput2
		convBackInput1 = multiply(self.reluDer1, reluBackInput1) #6*24*24
#		print "C1", convBackInput1
		self.conWeightGradient1 = self.conWeightGradient1 + self.weightGradConv(inputImg, convBackInput1) #6*5*5
		#print "CWG1", self.conWeightGradient1

	def gradZero(self):
		self.fcWeightGradient1.fill(0)
		self.fcWeightGradient2.fill(0)
		self.fcWeightGradient3.fill(0)
		self.conWeightGradient1.fill(0)
		self.conWeightGradient2.fill(0)

	def gradDecent(self, batchSize):

		self.fcWeight1 = self.fcWeight1 - self.fcWeightGradient1/batchSize
		self.fcWeight2 = self.fcWeight2 - self.fcWeightGradient2/batchSize
		self.fcWeight3 = self.fcWeight3 - self.fcWeightGradient3/batchSize
		self.convWeight1 = self.convWeight1 - self.conWeightGradient1/batchSize
		self.convWeight2 = self.convWeight2 - self.conWeightGradient2/batchSize



	def train(self, XTrain, YTrain, XValidate, YValidate, batchSize):
		trained = batchSize
		totalTrainingImages = len(XTrain)
		while trained < totalTrainingImages:
			self.gradZero()
			trained+= batchSize
			cost = 0
			imgIndexList = sample(range(0, totalTrainingImages), batchSize)
			for index in imgIndexList:
				cost += self.forwardFeed(XTrain[index], YTrain[index])
				self.backwardFeed(XTrain[index], YTrain[index])
			print "Cost ", trained/batchSize, " = ", cost/batchSize
			self.gradDecent(batchSize)

	def test(self, XTest, YTest):
		print "Testing"
		totalImages = len(XTest)
		correctPredictions = 0
		accuracy = 0
		for imgIndex in range(0, totalImages):
			cost = self.forwardFeed(X[imgIndex], Y[imgIndex])
			prediction = argmax(self.softMaxOut)
			print "Image index: ", imgIndex
			if prediction == Y[imgIndex][0]:
				correctPredictions+=1
			accuracy = correctPredictions*100.0/totalImages
		print "Accuracy: ", accuracy
		return accuracy

X,Y = loadMNIST('training')
XTest, YTest = loadMNIST('testing')
X = X/256.0
XTest = XTest/256.0
trainingData = zip(X, Y)
# random.shuffle(trainingData)
# X,Y = zip(*trainingData)
XTrain = X[:len(X)*3/4]
YTrain = Y[:len(X)*3/4]
XValidate = X[len(X)*3/4:]
YValidate = Y[len(X)*3/4:]
lenet = LENET5(32)
# # To plot backprop vs numercial gradient square sum
# # mlp.plotGradient(XTrain[:10], YTrain[:10])
lenet.train(XTrain, YTrain, XValidate, YValidate, 10)
lenet.test(XTest, YTest)
# lenet.forwardFeed(XTrain[0], YTrain[0])
# lenet.backwardFeed(XTrain[0], YTrain[0])
# lenet.gradDecent(1)


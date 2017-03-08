import math, os, struct
from array import array as pyarray
from pylab import *
from numpy import *
from random import sample
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from scipy import signal
import numpy

#todo: 32*32 

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
	images = zeros((N, rows, cols), dtype=uint8)
	labels = zeros((N, 1), dtype=int8)
	for i in range(len(ind)):
		images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape(rows,cols)
		labels[i] = lbl[ind[i]]
	return images, labels

def reluFunction(x):
	return max(0,x)

def reluDeriFunction(x):
	if x<=0:return 0
	else:return 1


class LENET5:

	def __init__(self, inputSize):
		self.relu = vectorize(reluFunction)
		self.reluDerivative = vectorize(reluDeriFunction)
		self.inputImg = zeros((28,28))
		self.convWeight1 = random.rand(20,5,5) * 0.001
		self.convWeight2 = random.rand(50,20, 5,5) * 0.001
		self.convOut1 = zeros((20,24,24))
		self.convOut2 = zeros((50,8,8))
		self.conWeightGradient1 = zeros((20,5,5))
		self.conWeightGradient2 = zeros((50, 20, 5, 5))
		self.reluOut1 = zeros((20,24,24))
		self.reluDer1 = zeros((20,24,24))
		self.reluOut2 = zeros((50,8,8))
		self.reluDer2 = zeros((50,8,8))
		self.reluOut3 = zeros((500))
		self.reluDer3 = zeros((500))
		self.poolOut1 = zeros((20,12,12))
		self.poolOut2 = zeros((50,4,4))
		self.poolMaxIndex1 = zeros((20,12,12))
		self.poolMaxIndex2 = zeros((50,4,4))
		self.fcWeight1 = random.rand(800,500) * 0.001
		self.fcWeightGradient1 =  zeros((800,500))
		self.fcOut1 = zeros(500)
		self.fcWeight2 = random.rand(500,10) * 0.001
		self.fcWeightGradient2 = zeros((500,10))
		self.fcOut2 = zeros((10))
		self.softMaxOut = zeros(10)

	def forwardConv(self, input, filters):
		output = zeros((filters.shape[0], input.shape[1]+1-filters.shape[2], input.shape[1]+1-filters.shape[2])) 
		for i in range(0, filters.shape[0]):
			if len(input.shape) == 3:
				for j in range(0, input.shape[0]):
					add(output[i], signal.convolve2d(input[j], rot90(rot90(filters[i][j])), mode= 'valid'))
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
				output[j] += signal.convolve2d(input[i], filters[i][j])
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
		self.inputImg = inputImg
		self.convOut1 = self.forwardConv(inputImg, self.convWeight1)
		self.reluOut1 = self.relu(self.convOut1)
		self.reluDer1 = self.reluDerivative(self.convOut1)
		self.poolOut1 = self.poolLayer(self.reluOut1, 2)
		self.poolMaxIndex1 = floor(divide(kron(self.poolOut1, ones((2,2))), self.reluOut1))
		self.convOut2 = self.forwardConv(self.poolOut1, self.convWeight2)
		self.reluOut2 = self.relu(self.convOut2)
		self.reluDer2 = self.reluDerivative(self.convOut2)
		self.poolOut2 = self.poolLayer(self.reluOut2, 2)
		self.poolMaxIndex2 = floor(divide(kron(self.poolOut2, ones((2,2))), self.reluOut2))
		self.fcOut1 = dot(self.poolOut2.reshape(1,800), self.fcWeight1)
		self.reluOut3 = self.relu(self.fcOut1)
		self.reluDer3 = self.reluDerivative(self.fcOut1)
		self.fcOut2 = dot(self.reluOut3, self.fcWeight2)
		self.softMaxOut = self.softmaxLayer(self.fcOut2.reshape(10))
		return -1*math.log(self.softMaxOut[lbl])		

	def backwardFeed(self,inputImg, lbl):
		self.softMaxOut[lbl] -= 1
		fcBackInput2 = self.softMaxOut #10
		self.fcWeightGradient2 += dot(self.reluOut3.reshape(500,1), fcBackInput2.reshape(1,10)) #500*10
		reluBackInput3 = dot(self.fcWeight2, fcBackInput2.reshape(10,1)) #500*1
		# print "ReluDR3 shape: ", self.reluDer3.shape
		# print "relubackinput3 shape:", reluBackInput3.shape
		fcBackInput1 = multiply(self.reluDer3.reshape(500,1), reluBackInput3) #500*1
		self.fcWeightGradient1 += dot(self.poolOut2.reshape(800,1), fcBackInput1.reshape(1,500)) # 800*500
		poolBackInput2 = dot(self.fcWeight1, fcBackInput1).reshape(50,4,4) # 50*4*4
		reluBackInput2 = multiply(kron(poolBackInput2, numpy.ones((1, 2,2))), self.poolMaxIndex2) #50*8*8
		convBackInput2 = multiply(self.reluDer2, reluBackInput2) #50*8*8
		self.conWeightGradient2 += self.weightGradConv(self.poolOut1, convBackInput2) #50*20*5*5
		poolBackInput1 = self.backpropConv(self.convOut2, self.convWeight2) #20*12*12
		reluBackInput1 = multiply(kron(poolBackInput1, numpy.ones((1, 2,2))), self.poolMaxIndex1) #20*24*24
		convBackInput1 = multiply(self.reluDer1, reluBackInput1) #20*24*24
		self.conWeightGradient1 += self.weightGradConv(inputImg, convBackInput1) #20*5*5

	def gradZero(self):
		self.fcWeightGradient1.fill(0)
		self.fcWeightGradient2.fill(0)
		self.conWeightGradient1.fill(0)
		self.conWeightGradient2.fill(0)

	def gradDecent(self, batchSize):
		self.fcWeight1 = self.fcWeight1 - self.fcWeightGradient1/batchSize
		self.fcWeight2 = self.fcWeight2 - self.fcWeightGradient2/batchSize
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
			print "Cost of ", trained/batchSize, "th iteration (total:", totalTrainingImages/batchSize, ") = ", cost/batchSize
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
random.shuffle(trainingData)
X,Y = zip(*trainingData)
XTrain = X[:len(X)*3/4]
YTrain = Y[:len(X)*3/4]
XValidate = X[len(X)*3/4:]
YValidate = Y[len(X)*3/4:]
lenet = LENET5(28)
# # To plot backprop vs numercial gradient square sum
# # mlp.plotGradient(XTrain[:10], YTrain[:10])
lenet.train(XTrain, YTrain, XValidate, YValidate, 10)
lenet.test(XTest, YTest)




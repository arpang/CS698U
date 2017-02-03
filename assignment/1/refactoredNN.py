import math, os, struct
from array import array as pyarray
from pylab import *
from numpy import *
from random import sample

def loadMNIST(dataset="training", digits=arange(10), path="."):
	# This function is taken from 
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

    images = zeros((N, rows * cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])
        labels[i] = lbl[ind[i]]
    return images, labels

class MLP:
	def __init__(self, nHiddenLayer, nNodes, actFunIndex, gradFunIndex):
		self.maxNode = int(max(nNodes))
		self.localGradient = zeros((nHiddenLayer+2, self.maxNode))
		self.weightList = random.rand(nHiddenLayer+1, self.maxNode, self.maxNode)
		self.weightList = self.weightList * 0.001 #0.0005, 70 percent
		self.weightGradient = zeros((nHiddenLayer+1, self.maxNode, self.maxNode))
		self.weightGradientRMSAverage = zeros((nHiddenLayer+1, self.maxNode, self.maxNode))
		self.weightGradientMomentum = zeros((nHiddenLayer+1, self.maxNode, self.maxNode))
		self.biasList = random.rand(nHiddenLayer+2, self.maxNode)
		self.biasList = self.biasList * 0.001
		self.biasGradient = zeros((nHiddenLayer+2, self.maxNode))
		self.biasGradientRMSAverage = zeros((nHiddenLayer+2, self.maxNode))
		self.biasGradientMomentum = zeros((nHiddenLayer+2, self.maxNode))
		self.nodeValue = zeros((nHiddenLayer+3, self.maxNode))
		self.backpropInput = zeros(self.maxNode)
		self.actFunIndex = actFunIndex
		self.gradFunIndex = gradFunIndex
		self.relu = 
		if actFunIndex!=0 and actFunIndex!=1:
			raise ValueError("index of activation function must be 0 or 1")
		if gradFunIndex!=0 and gradFunIndex!=1:
			raise ValueError("index of gradient descent function must be 0 or 1")

	def relu(self, x):
		return max(0,x)

	def reluDerivative(self, x):
		if x<=0: return 0
		else: return 1

	def tanh(self, x):
		return math.tanh(x)

	def tanhDerivative(self, x):
		return (1 - math.tanh(x)**2)

	def rmsProp(self, eta, gamma, epsilon = 1e-8):
		for i in range(0, nHiddenLayer+1):
			for j in range(0, int(nNodes[i])):
				for k in range(0, int(nNodes[i+1])):
					self.weightGradientRMSAverage[i][j][k] = gamma*self.weightGradientRMSAverage[i][j][k] + (1-gamma) * self.weightGradient[i][j][k] * self.weightGradient[i][j][k]
					self.weightList[i][j][k] -= eta*self.weightGradient[i][j][k]/sqrt(self.weightGradientRMSAverage[i][j][k]+epsilon)
				self.biasGradientRMSAverage[i][j] = gamma*self.biasGradientRMSAverage[i][j] + (1-gamma) * self.biasGradient[i][j] * self.biasGradient[i][j]
				self.biasList[i][j] -= eta*self.biasGradient[i][j]/sqrt(self.biasGradientRMSAverage[i][j]+epsilon)

		# # Second method
		# for i in range(0,nHiddenLayer+1):
		# 	j = int(nNodes[i])
		# 	k = int(nNodes[i+1])
		# 	self.weightGradientRMSAverage[i][:j][:,:k] = (0.1*gamma) * self.weightGradientRMSAverage[i][:j][:,:k]
		# 	self.weightGradientRMSAverage[i][:j][:,:k] += (1.0-gamma) * multiply(self.weightGradient[i][:j][:,:k], self.weightGradient[i][:j][:,:k]) 
		# 	self.weightList[i][:j][:,:k] = self.weightList[i][:j][:,:k] - eta*divide(self.weightGradient[i][:j][:,:k], sqrt(weightGradientRMSAverage[i][:j][:,:k]+epsilon)) 

		# 	self.biasGradientRMSAverage[i][:j] = (0.1*gamma) * self.biasGradientRMSAverage[i][:j]
		# 	self.biasGradientRMSAverage[i][:j] += (1.0-gamma) * multiply(self.biasGradient[i][:j], self.biasGradient[i][:j]) 
		# 	self.biasList[i][:j] = self.biasList[i][:j] - eta*divide(self.biasGradient[i][:j], sqrt(biasGradientRMSAverage[i][:j]+epsilon)) 
				

	def gdMomentum(self, eta, gamma):
		for i in range(0, nHiddenLayer+1):
			for j in range(0, int(nNodes[i])):
				for k in range(0, int(nNodes[i+1])):
					self.weightGradientMomentum[i][j][k] = gamma*self.weightGradientMomentum[i][j][k] + eta*self.weightGradient[i][j][k]
					self.weightList[i][j][k] -= self.weightGradientMomentum[i][j][k]
				self.biasGradientMomentum[i][j] = gamma*self.biasGradientMomentum[i][j] + eta*self.biasGradient[i][j]
				self.biasList[i][j] -= self.biasGradientMomentum[i][j]
		# # Second method
		# for i in range(0, nHiddenLayer+1):
		# 	j = int(nNodes[i])
		# 	k = int(nNodes[i+1])
		# 	self.weightGradientMomentum[i][:j][:,:k] = gamma*self.weightGradientMomentum[i][:j][:,:k] + eta*self.weightGradient[i][:j][:,:k]
		# 	self.weightList[i][:j][:,:k] = self.weightList[i][:j][:,:k] - self.weightGradientMomentum[i][:j][:,:k]
		# 	self.biasGradientMomentum[i][:j] = gamma*self.biasGradientMomentum[i][:j] + eta*self.biasGradient[i][:j]
		# 	self.biasList[i][:j] = self.biasList[i][:j] - self.biasGradientMomentum[i][:j]



	def softMax(self, inputArray): 
		output = copy(inputArray[:int(nNodes[nHiddenLayer+1])])
		exp = vectorize(math.exp)
		output = exp(output)
		total = sum(output)
		output = output/(1.0*total)
		# print "Softmax1:", output
		# output1 = zeros(int(nNodes[nHiddenLayer+1]))
		# total = 0
		# for i in range(0,int(nNodes[nHiddenLayer+1])):
		# 	output1[i] = math.exp(inputArray[i])
		# 	total += output1[i]
		# for i in range(0,int(nNodes[nHiddenLayer+1])):
		# 	output1[i] = output1[i]/total
		# print "Softmax2:", output1
		return output


	def costFunction(self, predictedLabel, actualLabel):
		# print actualLabel
		# print predictedLabel[:9]
		return -1*(math.log(predictedLabel[actualLabel[0]]))

	def forwardFeed(self, img, lbl,checkGradient=0):

		self.nodeValue[0] = copy(img)
		for i in range(1, nHiddenLayer+2):
			for j in range(0, int(nNodes[i])):
				iput = 0
				for k in range(0, int(nNodes[i-1])):
					iput+= self.nodeValue[i-1][k]*self.weightList[i-1][k][j]
				if self.actFunIndex==0:
					self.nodeValue[i][j] = self.relu(iput) + self.biasList[i][j]
					self.localGradient[i][j] = self.reluDerivative(iput)
				else:
					self.nodeValue[i][j] = self.tanh(iput) + self.biasList[i][j]
					self.localGradient[i][j] = self.tanhDerivative(iput)

		#second method
		# for i in range(1, nHiddenLayer+1):
		# 	inputArray = zeros((maxNode))
		# 	j = int(nNodes[i])
		# 	k = int(nNodes[i-1])
		# 	inputArray[:j] = dot(self.nodeValue[i-1][:k], self.weightList[i-1][:k][:,:j])
		# 	if self.actFunIndex==0:
		# 		actFun = vectorize(self.relu)
		# 		actFunDerivative = vectorize(self.reluDerivative)
		# 	else:
		# 		actFun = vectorize(self, tanh) 	
		# 		actFunDerivative = vectorize(self.tanhDerivative)

		# 	self.nodeValue[i] = actFun(inputArray[:j]) + self.biasList[i][:j]
		# 	self.localGradient[i] = actFunDerivative(inputArray[:j])


		i = nHiddenLayer+2
		self.nodeValue[i][:int(nNodes[i])] = self.softMax(self.nodeValue[i-1])
		return self.costFunction(self.nodeValue[i], lbl)

	def backwardFeed(self, lbl):
		self.backpropInput = copy(self.nodeValue[nHiddenLayer+2])
		self.backpropInput[lbl[0]] = self.backpropInput[lbl[0]] - 1
		i = nHiddenLayer + 1
		for j in range(0, int(nNodes[i])):
			self.localGradient[i][j] *= self.backpropInput[j]
		
		for i in range(nHiddenLayer, 0, -1):
			for j in range(0, int(nNodes[i])):
				self.backpropInput[j] = 0
				for k in range(0, int(nNodes[i+1])):
					self.backpropInput[j] += self.weightList[i][j][k]*self.localGradient[i+1][k]
				self.biasGradient[i][j] += self.backpropInput[j]
				self.localGradient[i][j] *= self.backpropInput[j]

		# second method
		# for i in range(nHiddenLayer,0,-1):
		# 	j = int(nNodes[i])
		# 	k = int(nNodes[i+1])
		# 	self.backpropInput[:j].fill(0)
		# 	self.backpropInput = dot(self.weightList[i][:j][:,:k], self.localGradient[i+1][:k])
		# 	self.localGradient[i][:j] = multiply(self.localGradient[i][:j], self.backpropInput[:j])
		# 	self.biasGradient[i][:j] = self.biasGradient[i][:j] +  self.backpropInput[:j]


		
		for i in range(1, nHiddenLayer+2):
			for j in range(0, int(nNodes[i])):
				for k in range(0, int(nNodes[i-1])):
					self.weightGradient[i-1][k][j] += self.localGradient[i][j]*self.nodeValue[i-1][k]

		#second method
		# for i in range(1, nHiddenLayer+2):
		# 	j = int(nNodes[i])
		# 	k = int(nNodes[i-1])
		# 	self.weightGradient[i-1][:k][:,:j] = self.weightGradient[i-1][:k][:,:j] + dot(self.nodeValue[i-1][:k], self.localGradient[i][:j])



	def train(self, XTrain, YTrain, XValidate, YValidate, iterations, nImage,):	

		totalImages = len(XTrain)
		print "Training"

		while iterations>0:
			iterations -=1		
			imgIndexList = sample(range(0, totalImages), nImage)

			# for i in range(1, nHiddenLayer+2):
			# 	for j in range(0, int(nNodes[i])):
			# 		for k in range(0, int(nNodes[i-1])):
			# 			self.weightGradient[i-1][k][j] = 0
			# 		self.biasGradient[i-1][k] = 0

			for i in range(1, nHiddenLayer+2):
				j = int(nNodes[i])
				k = int(nNodes[i-1])
				self.weightGradient[i-1][:k][:,:j].fill(0)
				self.biasGradient[i-1][:k].fill(0)

					
			cost = 0
			for imgIndex in imgIndexList:
				img = XTrain[imgIndex]
				lbl = YTrain[imgIndex]
				cost += self.forwardFeed(img, lbl)
				self.backwardFeed(lbl)

			print "Cost of ", iterations, "th iteration = ", cost/nImage
			for i in range(0, nHiddenLayer+1):
				for j in range(0, int(nNodes[i])):
					for k in range(0, int(nNodes[i+1])):
						self.weightGradient[i][j][k] /= nImage
					self.biasGradient[i][j]/=nImage

			#second method
			# for i in range(0, nHiddenLayer+1):
			# 	j = int(nNodes[i])
			# 	k = int(nNodes[i+1])
			# 	self.weightGradient[i][:j][:,:k] = self.weightGradient[i][:j][:,:k]/(1.0* nImage)
			# 	self.biasGradient[i][:j] = self.biasGradient[i][:j] /(1.0* nImage)
				
			if self.gradFunIndex==0:
				self.rmsProp(0.003,0.9)
			else:
				self.gdMomentum(0.003,0.9)

	def test(self, X, Y):
		# nHiddenLayer me bhi self lagana padega?
		print "Testing"
		totalImages = len(X)
		correctPredictions = 0
		accuracy = 0
		for imgIndex in range(0, totalImages):
			cost = self.forwardFeed(X[imgIndex], Y[imgIndex])
			prediction = argmax(self.nodeValue[nHiddenLayer+2][:int(nNodes[nHiddenLayer+2])])
			print "Image index: ", imgIndex
			if prediction == Y[imgIndex][0]:
				correctPredictions+=1
			accuracy = correctPredictions*100/totalImages
		print "Accuracy: ", accuracy
		return accuracy



nHiddenLayer = 1
nNodes = zeros(nHiddenLayer+4)
layer = 0
nNodes[layer] = 28*28
layer += 1
while layer<=nHiddenLayer:
	nNodes[layer] = 100
	layer+=1
nNodes[nHiddenLayer+1] = 10
nNodes[nHiddenLayer+2] = 10
nNodes[nHiddenLayer+3] = 1

X,Y = loadMNIST('training')
XTest, YTest = loadMNIST('testing')
X = X/256.0
XTest = XTest/256.0
XTrain = X[:len(X)*3/4]
YTrain = Y[:len(X)*3/4]
XValidate = X[len(X)*3/4:]
YValidate = Y[len(X)*3/4:]
mlp = MLP(nHiddenLayer, nNodes, 1, 0)
mlp.train(XTrain, YTrain, XValidate, YValidate, 1000, 10)
mlp.test(XTest, YTest)
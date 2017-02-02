import math, os, struct
from array import array as pyarray
from pylab import *
from numpy import *
from random import sample


actFunIndex_name = ''
nHiddenLayer = 1
nNodes = zeros(nHiddenLayer+4)



layer = 0
nNodes[layer] = 28*28
layer += 1
while layer<=nHiddenLayer:
	nNodes[layer] = 50
	layer+=1
nNodes[nHiddenLayer+1] = 10
nNodes[nHiddenLayer+2] = 10
nNodes[nHiddenLayer+3] = 1

actFunIndex = 1
mod = vectorize(mod)
maxNode = int(max(nNodes))
localGradient = zeros((nHiddenLayer+2, maxNode))
weightList = random.rand(nHiddenLayer+1, maxNode, maxNode)
weightList = mod(weightList, 0.00005)
weightGradient = zeros((nHiddenLayer+1, maxNode, maxNode))
weightGradientRMSAverage = zeros((nHiddenLayer+1, maxNode, maxNode))
biasList = zeros((nHiddenLayer+2, maxNode))
# biastList = mod(biasList, 0.005)
# biasGradient = zeros((nHiddenLayer+2, maxNode))
# biasGradientRMSAverage = zeros((nHiddenLayer+2, maxNode))
nodeValue = zeros((nHiddenLayer+3, maxNode))
backpropInput = zeros(maxNode)

def RMSProp(eta, epsilon, gamma):
	global weightGradientRMSAverage
	global weightList
	# global biasGradientRMSAverage
	# global biasList
	for i in range(0, nHiddenLayer+1):
		for j in range(0, int(nNodes[i])):
			for k in range(0, int(nNodes[i+1])):
				weightGradientRMSAverage[i][j][k] = gamma*weightGradientRMSAverage[i][j][k] + (1-gamma) * weightGradient[i][j][k] * weightGradient[i][j][k]
				weightList[i][j][k] -= eta*weightGradient[i][j][k]/sqrt(weightGradientRMSAverage[i][j][k]+epsilon)
			# biasGradientRMSAverage[i][j] = gamma*biasGradientRMSAverage[i][j] + (1-gamma) * biasGradient[i][j] * biasGradient[i][j]
			# biasList[i][j] -= eta*biasGradient[i][j]/sqrt(biasGradientRMSAverage[i][j]+epsilon)

def mod(x,d):
	return x%d

def softMax(array): 
	#print "Softmax input", array[:10]
	output = zeros(int(nNodes[nHiddenLayer+1]))
	total = 0
	for i in range(0,int(nNodes[nHiddenLayer+1])):
		output[i] = math.exp(array[i])
		total += output[i]
	for i in range(0,int(nNodes[nHiddenLayer+1])):
		output[i] = output[i]/total
	return output

def relu(x):
	return max(0,x)

def reluDerivative(x):
	if x<=0: return 0
	else: return 1

def tanh(x):
	return math.tanh(x)

def tanhDerivative(x):
	return (1 - math.tanh(x)**2)

def costFunction(predictedLabel, actualLabel):
	return -1*(math.log(predictedLabel[actualLabel[0]]))

def loadMNIST(dataset="training", digits=arange(10), path="."):
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


def mlpInit():
	global actFunIndex_name
	if(actFunIndex==0):
		actFunIndex_name = 'relu'
	elif(actFunIndex==1):
		actFunIndex_name = 'tanh'
	else:
		raise ValueError("index of activation function must be 0 or 1")

def forwardFeed(mlpInput, lbl, checkGradient=0):
	global nodeValue
	global weightList
	global backpropInput
	#global biasList
	global localGradient

	nodeValue[0] = copy(mlpInput)
	#print nodeValue[0]
	for i in range(1, nHiddenLayer+2):
		for j in range(0, int(nNodes[i])):
			iput = 0
			for k in range(0, int(nNodes[i-1])):
				iput+= nodeValue[i-1][k]*weightList[i-1][k][j]
			nodeValue[i][j] = globals()[actFunIndex_name](iput) + biasList[i][j]
			localGradient[i][j] = globals()[actFunIndex_name+'Derivative'](iput)

	i = nHiddenLayer+2
	nodeValue[i][:int(nNodes[i])] = softMax(nodeValue[i-1])
	return costFunction(nodeValue[i], lbl)

def backwardFeed(lbl):
	global nodeValue
	global weightList
	global weightGradient
	global backpropInput
	# global biasList
	# global biasGradient
	global localGradient
	backpropInput = copy(nodeValue[nHiddenLayer+2])
	backpropInput[lbl[0]] = backpropInput[lbl[0]] - 1
	i = nHiddenLayer + 1
	for j in range(0, int(nNodes[i])):
		localGradient[i][j] *= backpropInput[j]
	for i in range(nHiddenLayer, 0, -1):
		for j in range(0, int(nNodes[i])):
			backpropInput[j] = 0
			for k in range(0, int(nNodes[i+1])):
				backpropInput[j] += weightList[i][j][k]*localGradient[i+1][k]
			#biasGradient[i][j] += backpropInput[j]
			localGradient[i][j] *= backpropInput[j]
	
	for i in range(1, nHiddenLayer+2):
		for j in range(0, int(nNodes[i])):
			for k in range(0, int(nNodes[i-1])):
				weightGradient[i-1][k][j] += localGradient[i][j]*nodeValue[i-1][k]


def mlpTrain(trainingImgs, trainingLbls, iterations, nImage,):	
	global nodeValue
	global weightList
	global backpropInput
	global localGradient
	global weightGradient
	# global biasList
	# global biasGradient

	totalImages = len(trainingImgs)

	print "Training"
	while iterations>0:
		iterations -=1
		
		#print "Weight of weight[1][0][0] before", iterations, "th iteration:", weightList[0]
		imgIndexList = sample(range(0, totalImages), nImage)
		#print imgIndexList
		#global weightGradient = zeros((nHiddenLayer+1, maxNode, maxNode))
		weightGradient.fill(0)
		# biasGradient.fill(0)
		cost = 0
		for imgIndex in imgIndexList:
			img = trainingImgs[imgIndex]
			lbl = trainingLbls[imgIndex]
			cost += forwardFeed(img, lbl)
			backwardFeed(lbl)

		print "Cost of ", iterations, "th iteration = ", cost/nImage
		for i in range(0, nHiddenLayer+1):
			for j in range(0, int(nNodes[i])):
				for k in range(0, int(nNodes[i+1])):
					weightGradient[i][j][k] /= nImage
				#biasGradient[i][j]/=nImage
		RMSProp(0.003, 1e-8, 0.9)
			

def mlpTest(testingImgs, testingLbls):
	#global nodeValue

	print "Testing"
	totalImages = len(testingImgs)
	correctPrediction = 0
	for imgIndex in range(0, totalImages):
		cost = forwardFeed(testingImgs[imgIndex], testingLbls[imgIndex])
		# for layer in range(0, nHiddenLayer+3):
		# 	print "LAYER", layer, nodeValue[layer][:nNodes[layer]]
		prediction = argmax(nodeValue[nHiddenLayer+2][:int(nNodes[nHiddenLayer+2])])
		print "Image index: ", imgIndex, prediction, testingLbls[imgIndex][0]
		#print "Softmax output", nodeValue[nHiddenLayer+2][:int(nNodes[nHiddenLayer+2])] 
		if prediction == testingLbls[imgIndex][0]:
			correctPrediction+=1
			#print correctPrediction
	print "Accuracy: ", correctPrediction*100/totalImages





trainingImgs, trainingLbls = loadMNIST('training')
testingImgs, testingLbls = loadMNIST('testing')

mlpInit()
#print trainingImgs[0]
mlpTrain(trainingImgs, trainingLbls, 1000, 10)
mlpTest(testingImgs, testingLbls)

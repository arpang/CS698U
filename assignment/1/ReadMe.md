In in this assignment, I have implement a multilayer perceptron from scratch.

Programs takes the number of hidden layer, number of nodes in each hidden layer, gradient function index and activation function index. 
Number of nodes in input layer is set as 284 (=28*28)
After the hidden layers, there is an output layer with 10 nodes followed by a softmax layer with 10 nodes.
Cross entrophy cost function is used.


Below is the description of methods used:
__________________________________________

loadMNIST(): Loads and returns the MNIST image and label into numpy array

relu(x): Returns the value of ReLU(x)

reluDerivative(x): Returns the value of derivative of ReLU() at x

tanh(x): Returns the value of tanh(x)

tanhDerivative(x): Returns the value of derivative of tanh() at x

rmsProp(eta, gamma, epsilon): Applies RMSProp gradient descent to all weights and biases

gdMomentum(eta, gamma): Applies gradient descent with moment to all weights and biases 

softMax(array): Returns the output array after applying softmax function on input array

costFunction(predictedLabel, actualLabel): Returns the cross entropy cost function for the given predicted and actual label

forwardFeed(img, lbl): Feeds the img in the model and returns the value of cost function. Computes the node values and local gradient in the process.

backwardFeed(lbl): Backpropogation is implemeted in this function. After computing the local gradient descent for each neuron, gradient descent for all weights and biases are computed

train(XTrain, YTrain, XValidate, YValidate, totalIterations, nImage): In each of totalIterations iterations, a minibatch of size nImage is forward and backword fed to the network. Gradient is computed by taking the average of the minibatch. Gradient descent function then update each of the weight and bias  

test(testingImgs, testingImgLabels): Returns the accuracy for the given images and labels on the learned model

plotGradient(images, labels): For given images and labels, plots the square sum of numerical and backprop gradient for all parameters 

Description of graphs:
_______________________

I have plotted the training and validation error by varying the activation function, gradient descent function, number of neurons and number of hidden layer.

The graphs are plotted according to the following naming convention:
<Number of hidden layer>_<Number of neurons in each hidden layer>_<gradient descent function>_<activation function>_TrainError_ValidationError.

In addition, numerically computed squared sum of gradient of all parameters and that caculated during the back propogation step is also plotted.


Findings:
_________

1) On increasing the nodes in the hidden layer (50, 100, 200, 300) , accuracy increases
2) Tanh activation function performs better ReLU, keeping other parameters same. Using ReLU, the misclassification fraction is very jittery and learn is slow
3) Learning is slow initially in gradient descent with momentum as compared to RMSProp, but with large iterations both the models become comparable  
4) One hidden layer learns faster and performs better as compared to two hidden layers which performs better than three hidden layers (with number of neurons in each layer = 100)
5) With hidden layers = 1, neurons in hidden node = 300, tanh(x) activation function, RMSProp gradient descent and 5000 iterations, accuracy on test set is: 94.19%
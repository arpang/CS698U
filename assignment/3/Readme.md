Arpan Agrawal
Assignment 3

[1]

Numpy implementation:
________________________

Epochs: 1
Iterations:4500
Mini batch size: 10
Activation function: Leaky ReLU
Parameters: Only weights, no biases
Objective function: Cross entrophy
Optimizer: RMS Prop

Accuracy: 95.68%
Backpropogated gradients verified to be correct by numerically calculating gradients.

Tensorflow implementation:
____________________________

Epochs: 1
Iterations:4500
Mini batch size: 10
Activation function: ReLU
Parameters: Weights and biases both
Objective function: Cross entrophy
Optimizer: RMS Prop

Accuracy: 97.34%

Result in A1:
_______________

Architecture specs: Hidden layers = 1, Neurons in hidden layer = 300
Activation function: tanh(x)
Optimizer: RMSProp
Iterations: 5000 iterations
Batch size: 10
Accuracy: 94.33%

[2]

For one forward pass:
	Timt taken by conv layers:
	Time taken by fc layers:
	Ratio:

For one backward pass:
	Timt taken by conv layers:
	Time taken by fc layers:
	Ratio:

[3]

Number of parameters in conv layers:
Number of parameters in fc layers:
Ratio:

[4] 

Train_Valid_Error.png shows the plot of training and validation error vs. the number of iterations

[5]

Batch_Size.png plots the effect of different batch size on validation error
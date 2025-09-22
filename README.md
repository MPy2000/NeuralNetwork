# Neural Network/ Script

## Disclaimer

This is a modified and simple implementation of a neural Network inspired by the book "Make your own neural network" by Tariq rashid(German translation).
This script is a little more different to the original one in the book and it serves for educational purposes.

## Script

```python
# NeuralNetwork.py


import numpy             # numpy -> for matrix operations and numerical computations
import scipy.special     # scipy -> provides the sigmoid activation function


# Class definition for a simple feedforward neural network
class neuralNetwork: 

# store the number of nodes in each layer and the learning rate.
# self.lr: stores the learning rate for weight updates.
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):  
        self.inodes = inputnodes 
        self.hnodes = hiddennodes 
        self.onodes = outputnodes 

        # Initialize weight matrices with values between -0.5 and 0.5
        # Random values are generated using numpy's random number generator
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
        

        self.lr = learningrate 
        self.activation_function = lambda x: scipy.special.expit(x)  # sigmoid

    # Training method to adjust weights based on input and target output

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T   # data shape is (n, 1)
        targets = numpy.array(targets_list, ndmin=2).T  
        
        # forward pass
        # input is multiplied by weights, then passed through sigmnoid
        # Output is produced at both layers
         
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        # errors

        output_errors = targets - final_outputs # difference between target and prediction
        hidden_errors = numpy.dot(self.who.T, output_errors) # distributed back from output errors
        
        # backpropagation weight updates 
        # Gradient descent is used to update weights

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), 
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), 
                                        numpy.transpose(inputs))

    # performs only the forward pass 
    # Used to get preditcions after training 
    def query(self, input_list): 
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who , hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs 

# Small neural network for XOR problem
# XOR has 2 inputs and 1 output
# Hidden layer with 4 nodes to capture non-linear patterns
# Learning rate of 0.3 for weight updates

input_nodes = 2
hidden_nodes = 4 
output_nodes = 1
learning_rate = 0.3 

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Training data for XOR 
# Classic XOR truth table
training_data = [
    ([0,0], [0]),
    ([0,1], [1]),
    ([1,0], [1]),
    ([1,1], [0]),
]

# Train the network
# Each epoch processes all 4 XOR cases 
# Over time, the weights adjust the model XOR cases 

epochs = 5000   # Repeat training for 5000 iterations 
for e in range(epochs):
    for inputs, targets in training_data:
        n.train(inputs, targets)

# Test the network
print("Query results after training:")
print("0 XOR 0 =", n.query([0,0]))
print("0 XOR 1 =", n.query([0,1]))
print("1 XOR 0 =", n.query([1,0]))
print("1 XOR 1 =", n.query([1,1]))

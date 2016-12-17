'''
    These are just the commands to run a basic example of the neural network
    according to chapter 1 of neuralnetworksanddeeplearning.com.
'''
import dataloader
import network

training_data, validation_data, test_data = dataloader.load_wrapper()
inputNodeNum = 784 # One for every character of the 28 x 28 digit data
hiddenNodeNum = 30 # Number of nodes in the hidden layer
outputNodeNum = 10 # One for every class that a digit could be
layers = [inputNodeNum, hiddenNodeNum, outputNodeNum]



# Parameters for stochastic gradient descent.
''' 
     The batch of digit data is broken down into mini batches, so that we can randomize the order of digit training.
'''
mini_batch_size = 10 

'''
    The number of iterations to train the data. For every iteration, a new configuration of mini batches is created, and then the neural network is 
    trained on that new configuration. This means that in every iteration, ALL of the training data is used, except the order of digits chosen to train
    is different between iterations.

    If test_data was specified, after each iteration the network is evaluated.
'''
iterations = 30 
learningrate = 3.0 # The 'speed' at which this network learns. 

nnet = network.Network(layers)
nnet.SGD(training_data, iterations, mini_batch_size, learningrate, test_data=test_data)
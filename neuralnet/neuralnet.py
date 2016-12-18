import dloader
import network

print("What kind of data? ('1' for digits, '0' for faces)")
q = int(raw_input())
isdigit = True
if q == 0:
    isdigit = False
print("isDigit: " + str(isdigit))

print("Training data size: ")
tdn = int(raw_input())

print("Validation data size: ")
vdn = int(raw_input())

print("Test data size: ")
tedn = int(raw_input())

training_data, validation_data, test_data = dloader.load_wrapper(tdn, vdn, tedn, isdigit)

print("Number of input nodes (should be = height * width of image): ")
inn = int(raw_input())

hiddenlayernum = 0
hiddenlayers = []
while True:
    print("Number of nodes for hidden layer " + str(hiddenlayernum) + " (enter -1 to finish): ")
    num = int(raw_input())
    if num == -1: break
    hiddenlayers.append(num)
    hiddenlayernum += 1

print("Number of output nodes (should be = number of classes possible; 10 for digits, 2 for faces): ")
outn = int(raw_input())

layers = [inn] + hiddenlayers + [outn]

print("Network configuration: " + str(layers))


# Parameters for stochastic gradient descent.
''' 
     The batch of data is broken down into mini batches, so that we can randomize the order of training.
'''
print("Mini-batch size: ")
mini_batch_size = int(raw_input()) 

'''
    The number of iterations to train the data. For every iteration, a new configuration of mini batches is created, and then the neural network is 
    trained on that new configuration. This means that in every iteration, ALL of the training data is used, except the order of digits chosen to train
    is different between iterations.

    If test_data was specified, after each iteration the network is evaluated.
'''
print("Iterations of training: ")
iterations = int(raw_input())

print("Learning rate: ") 
learningrate = float(raw_input()) # The 'speed' at which this network learns. 

nnet = network.Network(layers)
nnet.SGD(training_data, iterations, mini_batch_size, learningrate, test_data=test_data)
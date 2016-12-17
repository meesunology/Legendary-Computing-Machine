import numpy
import random

class Network():
    """
    Neural Network implementation.
    See http://neuralnetworksanddeeplearning.com/chap1.html
    """

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        # the first layer is the input layer, so no biases or weights to set there
        self.biases = [numpy.random.randn(y, 1) for y in sizes [1:]]
        self.weights = [numpy.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        
    def sigmoid(self, z):
        return 1.0/(1.0 + numpy.exp(-z))
    
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1-self.sigmoid(z))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(numpy.dot(w, a) + b)
        return a

    def SGD(self, training_data, iterations, mini_batch_size, learningrate, test_data=None):
        '''
        Train this network using Stochastic Gradient Descent. 
        '''
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(iterations):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)
            ]
            for batch in mini_batches:
                self.update_mini_batch(batch, learningrate)
            if test_data:
                print "Iteration {0} : {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Iteration {0} complete".format(j)
    
    def update_mini_batch(self, mini_batch, learningrate):
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        '''
        self.weights = [w-(learningrate/len(mini_batch)) * nw 
        for w, nw in zip(self.weights, nabla_w)]
        
        self.biases - [b-(learningrate/len(mini_batch)) * nb 
        for b, nb in zip(self.biases, nabla_b)]
        '''

        self.weights = [w-(learningrate/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learningrate/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        """
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        
        activation = x
        activations = [x] # list of all activations, by layer
        zs = [] # list of all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = numpy.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(numpy.argmax(self.feedforward(x)), y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)

    def cost_derivative(self, output_activations, y):
        return(output_activations - y)

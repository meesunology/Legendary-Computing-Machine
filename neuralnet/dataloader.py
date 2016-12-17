import cPickle
import gzip
import zipfile
import os

import numpy as np

def load():
    """
    Returns digit data as <training data, validation data, test data>, where
    training data = <training_images, training_labels>
        training_images = numpy ndarray of 50,000 entries
            each entry is a numpy ndarray of 784 values, corresponding to pixels of images
        training_labels = numpy ndarray of 50,000 labels corresponding to above
            each entry is just a digit value labels
    validation data and test data are the same thing, except with only 10,000 each
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_wrapper():
    """
    Makes the format of load() easier to work with neural networks. 
    training_data = list of 50,000 2-tuples (x,y), 
        where x is a 784d float numpy.ndarray that has the input image,
        and y is a 10d float numpy.ndarray unit vector for the correct digit for x that is returned by vectorized_result()
    validation_data and test_data = list of 10,000 2-tuples (x,y),
        where x is a 784d numpy.ndarray with the input image,
        and y = the actual digit values (integers) corresponding to x
    """
    tr_d, va_d, te_d = load()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
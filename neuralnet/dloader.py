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

# Raw data loading
# loads n digit datas
def dataload(filename, width, height, n):
    '''
        Loads a data file containing raw ASCII art images, with each piece of data being of the specified width and height.
        Returns a list of vectors, where one entry is a (height*width,1) vector of the characters of each data piece.
    '''
    if(os.path.exists(filename)):
        with open(filename, 'r') as dataFile:
            raw = [x[:-1] for x in dataFile.readlines()]
            raw.reverse()
            data = []
            for i in range(n):
                image = []
                empty = False
                for j in range(height):
                    if raw:
                        image = image + list(raw.pop())
                    else:    
                        print("Truncating at " + str(i) + " examples from " + filename)
                        empty = True
                        break
                if empty: break
                data.append(image)
            darray = np.array(data)
            darray = [np.reshape(x, (height*width,1)) for x in darray] # length of this array is n
            return extractfeatures(darray)
    else:
        print(filename + " not found in this directory.")
        return None

def labelload(filename, n, vectorize = False):
    if(os.path.exists(filename)):
        with open(filename, 'r') as dataFile:
            raw = [x[:-1] for x in dataFile.readlines()]
            raw.reverse()
            labels = []
            for i in range(n):
                if raw:
                    num = int(raw.pop())
                    labels.append(num)
                else:
                    print("Truncating at " + str(i) + " labels from " + filename)
                    break
            if vectorize: return [vectorized_result(y) for y in labels]
            else: return labels
    else:
        print(filename + " not found in this directory.")
        return None

def learningdataload(datafile, width, height, labelfile, size, vectorize=False):
    '''
        Returns a tuple of the <training_data, training_labels>
    '''
    return zip(dataload(datafile, width, height, size), labelload(labelfile, size, vectorize))

def extractfeatures(data, blankf = 0.0, plusf = 0.5, octof = 1.0):
    '''
        Takes an ndarray of vectors, each being a (height*width, 1) ndarray of the characters ' ', '+', or '#',
        and returns a similar structure with float values from 0 to 1.

        Default:
        ' ' = 0.0
        ' ' = 0.5
        ' ' = 1.0
    '''
    features = []
    for i in range(len(data)):
        vector = []
        for j in data[i]:
            if j == '+': val = plusf
            elif j == '#': val = octof
            else: val = blankf
            vector.append([val])
        features.append(vector)
    vector = np.array(features)
    reshaped = [np.reshape(x, data[0].shape) for x in features]
    return reshaped

def load_wrapper(tr_size, va_size, te_size, digits=True):
    width = 0
    height = 0
    datadirectory = os.path.dirname(__file__)
    filepath = ''
    if digits:
        width = 28
        height = 28
        filepath = os.path.join(datadirectory, 'data', 'digitdata')
    else:
        width = 60
        height = 70
        filepath = os.path.join(datadirectory, 'data', 'facedata')
    
    ti = os.path.join(filepath, 'trainingimages')
    tl = os.path.join(filepath, 'traininglabels')

    vi = os.path.join(filepath, 'validationimages')
    vl = os.path.join(filepath, 'validationlabels')

    ei = os.path.join(filepath, 'testimages')
    el = os.path.join(filepath, 'testlabels')


    tr_d = learningdataload(ti, width, height, tl, tr_size, digits)
    va_d = learningdataload(vi, width, height, vl, va_size)
    te_d = learningdataload(ei, width, height, el, te_size)

    return (tr_d, va_d, te_d)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
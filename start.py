import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab
import sklearn
import sklearn.svm
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

import csv
import pickle
import sys
import random

# serialized object files for the data matrix and labels
TRAIN_DATA_PICKLE = 'train_data.pickle'
TEST_DATA_PICKLE = 'test_data.pickle'

# source CSV files from Kaggle
TRAIN_DATA_CSV = 'train.csv'
TEST_DATA_CSV = 'test.csv'

# eigenvector files
EIGENVECTOR_COVX_FIRST_2500 = 'eigs_x_2500.pickle'

# prediction files
KMEANS_RANDOM_INIT = "kmeans_random_predictions.txt"
KMEANS_MEAN_INIT = "kmeans_mean_predictions.txt"

def savePickle(obj, filename):
    """
    Save this object as a pickled object file.
    """
    with open(filename, 'w') as f:
        pickle.dump(obj, f)
    f.close()

def loadPickle(filename):
    """
    Load this pickle file from a particular path location.
    """
    with open(filename, 'r') as f:
        return pickle.load(f)

def l2(x1, x2):
    return np.sum(np.square(x1 - x2))

def csv2matrix(csvpath, hasFirstRowAsLabels=True, pickleFile=None):
    """
    Reads a CSV file into a matrix. Also optionally saves the data matrix
    into a pickled file and extracts header labels for fields. 
    """
    cols = -1
    i = 0
    featureNames = list()

    # read the file, then preallocate for speed
    with open(csvpath, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            
            # save the feature names
            if hasFirstRowAsLabels and cols == -1:
                featureNames = np.array(row)
                cols = len(row)
                continue
            else:
                i += 1

    # allocate memory for the array
    data = np.matrix(np.zeros((i, cols)))
    i = 0
    gotHeader = not hasFirstRowAsLabels

    # then create the matrix
    with open(csvpath, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:

            if not gotHeader:
                gotHeader = True
                continue

            # add to our matrix
            r = [float(elt) for elt in row]

            data[i, :] = np.matrix(r)
            i += 1

    # should we serialize the object and save to disk?
    if pickleFile:
        if hasFirstRowAsLabels:
            savePickle({'data' : data, 'names' : featureNames}, pickleFile)
        else:
            savePickle({'data' : data}, pickleFile)

    if hasFirstRowAsLabels:
        return data, featureNames

    return data

def kMeans(X, init, metric):
    
    # get value of k and n
    k = init.shape[0]
    n = X.shape[0]
    d = init.shape[1]

    # vars to reuturn
    clusterAssignments = np.matrix(np.zeros((n, 1)))

    # error vars
    error = sys.maxint
    errorDiff = sys.maxint

    # while we are still decreasing the error
    while (errorDiff > 0):

        ### readjust cluster membership
        for i in range(n):

            minDist = sys.maxint

            # and each cluster
            for j in range(k):

                # change this point's cluster if need be
                distToCentroid = metric(np.copy(X[i, :]), np.copy(init[j, :][:]))
                if distToCentroid < minDist:
                    minDist = distToCentroid
                    clusterAssignments[i, 0] = j

        # make counter and accumulator for points to recalculate means
        centroidAccumulator = np.matrix(np.zeros((k, d)))
        centroidMembershipCounter = {}
        for j in range(k):
            centroidMembershipCounter[j] = 0

        ### recalculate means
        for i in range(n):

            # get our data row
            row = np.matrix(np.copy(X[i, :]))

            # get the cluster this example is curently associated with
            rowMembership = clusterAssignments[i, 0]

            # add each example to appropriate row of accumulator and count membership
            centroidAccumulator[rowMembership, :] += row
            centroidMembershipCounter[rowMembership] += 1

        # then take average
        for j in range(k):
            centroidAccumulator[j, :] /= centroidMembershipCounter[j]

        # copy over our new centroids to init
        init = np.copy(centroidAccumulator)

        ## calculate error
        cumulativeError = 0
        for i in range(n):
            assignment = clusterAssignments[i, 0]
            assignmentCentroid = init[assignment, :]
            cumulativeError += metric(np.copy(X[i, :]), np.copy(assignmentCentroid))

        # update our error counters
        errorDiff = error - cumulativeError
        error = cumulativeError

    return (init, clusterAssignments)

def mlCovarianceMatrix(X):

    # get the dimensions
    n = X.shape[0]
    d = X.shape[1]

    # calculate means
    means = np.zeros((1, d))
    for feature in range(d):
        means[0, feature] = np.mean(X[:, feature])

    # center the data
    for i in range(n):
        X[i, :] = X[i, :] - means

    # create covariance matrix
    covarianceMatrix = np.zeros((d, d))
    for i in range(n):
        covarianceMatrix = covarianceMatrix + np.matrix(X[i, :]).T * np.matrix(X[i, :]) 

    return covarianceMatrix / n

# Perform PCA, optionally apply the "sphering" or "whitening" transform, in
# which each eigenvector is scaled by 1/sqrt(lambda) where lambda is
# the associated eigenvalue.  This has the effect of transforming the
# data not just into an axis-aligned ellipse, but into a sphere.  
# Input:
# - X: n by d array representing n d-dimensional data points
# Output:
# - u: d by n array representing n d-dimensional eigenvectors;
#      each column is a unit eigenvector; sorted by eigenvalue
# - mu: 1 by d array representing the mean of the input data
# -  l: list of non-zero eigenvalues
# This version uses SVD for better numerical performance when d >> n

def PCA(X, sphere = False):
    (n, d) = X.shape
    mu = np.mean(X, axis=0)
    (x, l, v) = np.linalg.svd(X-mu)
    l = np.hstack([l, np.zeros(v.shape[0] - l.shape[0], dtype=float)])
    u = np.array([vi/(li if (sphere and li>1.0e-10) else 1.0) \
                  for (li, vi) \
                  in sorted(zip(l, v), reverse=True, key=lambda x: x[0])]).T
    return np.matrix(u), mu

def pca(X):
    """
    Steps:
        1) Create covariance matrix of the data
        2) Extract eigenvalues and eigenvectors
        3) Sort by size of eigenvalues, remove any below 1e-10
    """
    
    # create (d x d) covariance matrix
    covX = mlCovarianceMatrix(X)

    # extract eigens
    eigenvalues, eigenvectors = np.linalg.eig(covX)

    # then sort in ascending order
    eig_vals_sorted = np.sort(eigenvalues)[::-1]
    eig_vecs_sorted = np.matrix(np.fliplr(eigenvectors[eigenvalues.argsort()]))

    # need to remove small (eigenvalues < 1e-10) eigenvectors
    toDel = []
    for i in range(len(eig_vals_sorted)):
        if eig_vals_sorted[i] <= 1e-10:
            toDel.append(i)

    return np.delete(eig_vecs_sorted, toDel, 0).T

'''
def projection(dataX, eigensCovX, numComponents):
    """
    Returns a reduced dimensionality version of X
    by approximating the matrix

    X             = original (n x d) data matrix
    eigensCovX    = covariance feature matrix of X such that it is (d x d)
    numComponents = number of componenets we will use to approximate X
                    scalar in range [d, n]
    """
    X = np.copy(dataX)
    n = X.shape[0]

    # subtract out the mean
    p = np.mean(X, axis = 0)
    for i in range(n):
        X[i, :] -= p

    # then project using only first componenets
    return X * eigensCovX[:, 0:numComponents]

def reconstruction(originalX, eigensCovX, projectedX):
    """
    Reconstructs the data given the reduced dimensional data

    originalX  = original (n x d) data matrix, we need it only for the mean to uncenter the data
    eigensCovX = all eigenvectors (sorted) of covariance of X (d x d)
    projectedX = the reduced dimensionality version of X (m examples x c dimensions)
    """

    print "Reconstructing..."
    print "originalX shape:", originalX.shape
    print "eigensCovX shape:", eigensCovX.shape
    print "projectedX shape:", projectedX.shape

    # grab c
    c = projectedX.shape[1]
   
    # multiply the reduced eigenvalue matrix (c eigenvectors)
    # with the transpose of the projectedX matrix, which is the data
    # with its dimensionality reduced by projection onto the
    # c largest eigens
    subsetE = np.matrix(eigensCovX[:, 0:c])

    print "subsetE shape:", subsetE.shape

    reconstructed = np.matrix(projectedX) * subsetE.T 
    
    # find the origin (mean) and add it back in
    p = np.mean(originalX, axis = 0)
    m = reconstructed.shape[0]
    for i in range(m):
        reconstructed[i, :] += p
        
    return reconstructed
'''

def projection(X, E, l):

    # number of classes
    numClasses = X.shape[0]

    # find the origin (mean) and subtract it out
    p = np.mean(X, axis = 0)
    for i in range(numClasses):
        X[i, :] = X[i, :] - p

    # extract 
    reducedE = np.matrix(E[:, 0:l])

    # then multiply
    return np.matrix(X) * reducedE

def reconstruction(X, E, P):

    # grab c
    c = P.shape[1]
   
    # multiply the reduced eigenvalue matrix (c eigenvectors)
    # with the transpose of the P matrix, which is the data
    # with its dimensionality reduced by projection onto the
    # c largest eigens
    subsetE = np.matrix(E[:, 0:c])
    answer = np.matrix(P) * subsetE.T 
    
    # find the origin (mean) and add it back in
    p = np.mean(X, axis = 0)
    m = answer.shape[0]
    
    for i in range(m):
        answer[i, :] = answer[i, :] + p
        
    return answer

_image_size = (28, 28)
def showIm(im, size = _image_size):
    """
    im: a row or column vector of dimension d
    size: a pair of positive integers (i, j) such that i * j = d
       defaults to the right value for our images
    """
    plt.figure()
    im = im.copy()
    im.resize(*size)
    plt.imshow(im.astype(float), cmap = cm.gray)
    pylab.show()

# Take an eigenvector and make it into an image
def vecToImage(x, size = _image_size):
  im = x/np.linalg.norm(x)
  im = im*(256./np.max(im))
  im.resize(*size)
  return im

def plotMeanByClass(X, y):
    meansDict = {}
    means = []

    # size vars
    uniques = np.unique(np.array(y))
    n_classes = len(uniques)
    n = X.shape[0]
    d = X.shape[1]

    # allocate room for mean indices lists
    for i in range(n_classes):
        meansDict[i] = []
        means.append(np.matrix(np.zeros((1, d))))

    # collect all the indices for certain classes
    for i in range(n):
        label = y[i][0]
        if label in meansDict:
            meansDict[label].append(i)

    # find mean and plot for each digit
    for i in range(n_classes):

        rows = []
        for j in meansDict[i]:
            rows.append(X[j, :])

        # stack them together and average
        # then show name of target and plot
        stacked = np.vstack(rows)
        means[i] = np.mean(stacked, axis=0)
        
        # to display the means
        #print "Now showing mean of %s" % str(uniques[i])
        #showIm(means[i])

    # make into a matrix and return
    # each row is the character class [0-9]
    return np.vstack(means)

def kMeansFit(X, centers, metric):
    """
    X       = (n x d) data matrix
    centers = (k x d) cluster centers
    """

    # get size variables
    n = X.shape[0]
    k = centers.shape[0]
    predictions = np.matrix(np.zeros((n, 1)))

    # for each test point
    for i in range(n):

        minDist = sys.maxint
        minCluster = -1
        example = X[i, :]

        # for each potential cluster
        for j in range(k):
            dist = metric(centers[j], example)
            if dist < minDist:
                minDist = dist
                minCluster = j

        # assign the best one we found
        predictions[i, 0] = minCluster

    return predictions

def digitRecognitionKMeans(meanInit=True, loadPickledData=True):
    """
    Performs kMeans clustering on the digits dataset with options for 
    initialization and whether to load from previously calculated data 
    matrix from disk.
    """

    if meanInit:
        print
        print "Starting kMeans clustering with mean digit initialization"
    else:
        print 
        print "Starting kMeans clustering with random digit initialization"

    #### TRAIN ####
    # load the training data
    trainX = None
    if loadPickledData:
        print "[*] Loading data matrix..."
        data = loadPickle(TRAIN_DATA_PICKLE)
        X = data['data']
    else:
        print "[*] Converting data from CSV to matrix format..."
        X, names  = csv2matrix(TRAIN_DATA_CSV, True, TRAIN_DATA_PICKLE)
    
    # ready it for processing
    Y = np.array(X[:, 0])
    X = X[:, 1:]

    # split into training and testing
    (trainX, testX, trainY, testY) = sklearn.cross_validation.train_test_split(X, Y, test_size=0.75)

    # try clustering by means
    init = None
    if meanInit:
        print "[*] Computing means of training dataset digits..."
        init = plotMeanByClass(trainX, trainY)
    else:
        # do random init
        print "[*] Computing random centers for initialization..."
        init = np.random.random_integers(0, high=256, size=(10, 784))
    
    print "[*] Building clustering model..."
    centers, clusterAssignments = kMeans(trainX, init, l2)

    # get training score
    trainingPredictions = kMeansFit(trainX, centers, l2)
    accuracy = np.sum(trainingPredictions == trainY) / float(trainY.shape[0])
    print "Training accuracy: %f" % accuracy

    # get testing score
    testingPredictions = kMeansFit(testX, centers, l2)
    accuracy = np.sum(testingPredictions == testY) / float(testY.shape[0])
    print "Testing accuracy: %f" % accuracy

    # clear memory 
    print "[*] Clearing memory of training & testing data..."
    del trainX, trainY, testX, testY, data

    ### TEST ###
    if loadPickledData:
        print "[*] Loading validation test data set..."
        data = loadPickle(TEST_DATA_PICKLE)
        testX = data['data']
    else:
        print "[*] Converting data from CSV to matrix format..."
        testX, names = csv2matrix(TEST_DATA_CSV, True, TEST_DATA_PICKLE)

    # use the model to fit
    print "[*] Fitting to kMeans model..."
    predictions = kMeansFit(testX, centers, l2)

    # then write the predictions to disk
    print "[*] Writing predictions to disk..."
    filename = None
    if meanInit:
        filename = KMEANS_MEAN_INIT
    else:
        filename = KMEANS_RANDOM_INIT
    f = open(filename, 'w')
    for i in range(predictions.shape[0]):
        f.write(str(predictions[i, 0]) + "\n")

    # clear memory
    print "[*] Clearing memory of test data..."
    del testX, data, predictions

def kMeansWithPCA(meanInit=True):
    """
    RESULTS (with 2500 examples):

    accuracies = [0.1708, 0.2524,0.3684,0.4344,0.5332,0.5924,0.7204,0.7392,0.734,0.7492,0.7476,0.7476]
    l_values = [1, 5, 15, 50, 100, 200, 300, 400, 500, 600, 700, 784]
    """

    # try different values of number of components
    l_values = [1, 5, 15, 50, 100, 200, 300, 400, 500, 600, 700, 784]
    accs = []
    for l in l_values:
        accs.append(pcatest(l, meanInit))

    # then plot the resulting curve to see the optimal values for PCA
    plt.figure('Accuracy as a function of number of components projected')
    ax = plt.subplot(111)
    ax.plot(l_values, accs, label = "Accuracy")
    plt.xlabel('Number of components (l)')
    plt.ylabel('Accuracy')
    ax.legend(loc='lower right')
    plt.show()

def pcatest(l, meanInit=True):
    """
    Does a single run of projecting training examples into the lower dimensional space,
    running kMeans with the means initialized 
    """
    print
    print "[*] Loading data..."
    data = loadPickle(TRAIN_DATA_PICKLE)
    X = data['data']
    Y = np.array(X[:, 0])
    X = X[:, 1:]

    """ Comment this in to restrict the size of the dataset
    X = X[0:2500, :]
    Y = Y[0:2500, :]
    """

    print "[*] Loading eigenvectors..."
    E = loadPickle(EIGENVECTOR_COVX_FIRST_2500)

    # initialize clusters
    init = None
    if meanInit:
        print "[*] Computing means of training dataset digits..."
        init = plotMeanByClass(X, Y)
    else:
        # do random init
        print "[*] Computing random centers for initialization..."
        init = np.random.random_integers(0, high=256, size=(10, l))

    print "[*] Projecting examples down into (%d) dimensional space..." % l
    projX = projection(X, E, l)
    projMeanX = projection(init, E, l)

    print "[*] Building clustering model..."
    centers, clusterAssignments = kMeans(projX, projMeanX, l2)

    # get training score
    trainingPredictions = kMeansFit(projX, centers, l2)
    accuracy = np.sum(trainingPredictions == Y) / float(Y.shape[0])
    print "[*] Training accuracy for %d components: %f" % (l, accuracy)
    return accuracy

'''
############################ RESULTS ########################################
# Training accuracy: 0.103714
# Testing accuracy: 0.108794
digitRecognitionKMeans(meanInit=False, loadPickledData=True)

# Training accuracy: 0.762571
# Testing accuracy: 0.756635
# kaggle validation: 0.75
digitRecognitionKMeans(meanInit=True, loadPickledData=True)
'''

kMeansWithPCA(True)

kMeansWithPCA(False)

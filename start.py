import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab

import sklearn
import sklearn.svm
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, average_precision_score, accuracy_score

import csv
import pickle
import sys
from heapq import *

# image dimensions
IMG_HEIGHT    = 28
IMG_WIDTH     = 28
IMG_DIMENSION = IMG_HEIGHT * IMG_WIDTH
IMG_SIZE      = (IMG_HEIGHT, IMG_WIDTH)

# data characteristics
NUM_CLASSES = 10 # [0-9]

# training / testing params
TEST_RATIO = 0.25

# serialized object files for the data matrix and labels
TRAIN_DATA_PICKLE           = "train_data.pickle"
TEST_DATA_PICKLE            = "test_data.pickle"
EIGENVECTOR_COVX_FIRST_2500 = "eigs_x_2500.pickle"
EIGENVECTOR_COVX_ALL        = "eigs_x_all.pickle"

# source CSV files from Kaggle
TRAIN_DATA_CSV = "csv/train.csv"
TEST_DATA_CSV  = "csv/test.csv"

# prediction files
KMEANS_RANDOM_INIT = "kmeans_random_predictions.txt"
KMEANS_MEAN_INIT   = "kmeans_mean_predictions.txt"

# directories
KNN_DIR     = "plots/kNN"
KMEANS_DIR  = "plots/kMeans"
MEANS_DIR   = "plots/means"
PICKLES_DIR = "pickles"

def savePickle(obj, filename):
    """
    Save this object as a pickled object file.
    """
    filename = "%s/%s" % (PICKLES_DIR, filename)
    with open(filename, 'w') as f:
        pickle.dump(obj, f)
    f.close()

def loadPickle(filename):
    """
    Load this pickle file from a particular path location.
    """
    filename = "%s/%s" % (PICKLES_DIR, filename)
    with open(filename, 'r') as f:
        return pickle.load(f)

def l2(x1, x2):
    return np.sum(np.square(x1 - x2))

def getTrainData(loadPickledData=True):
    """
    Loads or calculates the training data and labels from disk. 
    """

    if loadPickledData:
        print "[*] Loading training data matrix..."
        data = loadPickle(TRAIN_DATA_PICKLE)
        X = data['data']
    else:
        print "[*] Converting training data from CSV to matrix format..."
        X, names  = csv2matrix(TRAIN_DATA_CSV, True, TRAIN_DATA_PICKLE)
    
    # ready it for processing
    Y = np.array(X[:, 0])
    X = X[:, 1:]
    return (X, Y)

def getTestData(loadPickledData=True):
    """
    Loads or calculates the test data and labels from disk. 
    """

    if loadPickledData:
        print "[*] Loading test data matrix..."
        data = loadPickle(TEST_DATA_PICKLE)
        X = data['data']
    else:
        print "[*] Converting test data from CSV to matrix format..."
        X, names  = csv2matrix(TEST_DATA_CSV, True, TRAIN_DATA_PICKLE)
    
    # ready it for processing
    Y = np.array(X[:, 0])
    X = X[:, 1:]
    return (X, Y)

def csv2matrix(csvpath, hasFirstRowAsLabels=True, pickleFileName=None):
    """
    Reads a CSV file into a matrix. Also optionally saves the data matrix
    into a pickled file and extracts header labels for fields. 

    if no pickleFileName is specified (is None), the data is not saved to disk
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
    if pickleFileName:
        if hasFirstRowAsLabels:
            savePickle({'data' : data, 'names' : featureNames}, pickleFileName)
        else:
            savePickle({'data' : data}, pickleFileName)

    if hasFirstRowAsLabels:
        return data, featureNames

    return data

def kMeans(X, init, metric):
    """
    The kMeans algorithm. Provided an initialization for the cluster
    centroids and the dataset X, we use the distance metric to continue iteration
    untill the distance classification error ceases decreasing.

    For the l2 metric, the kMeans algorithm gurantees non-increasing error 
    through iterations.
    """
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
    """
    Calculate the covariance matrix of X (n x d) as a 
    (d x d) matrix. 
    """
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

def pca(X, loadFromPickle=False):
    """
    Steps:
        1) Create covariance matrix of the data
        2) Extract eigenvalues and eigenvectors
        3) Sort by size of eigenvalues, remove any below 1e-10
    """

    if loadFromPickle:
        return loadPickle(EIGENVECTOR_COVX_ALL)
    
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

def projection(X, E, l):
    """
    Projects the dataset X into a lower dimensional (l) space
    using E, the eigenvectors of covariance(X).
    """
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
    """
    Reconstructs the projected dataset P by using E, the 
    eigenvectors of covariance(X). The original X is required only 
    for calculating the mean to uncenter the data. 
    """
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

def showIm(im, size = IMG_SIZE):
    """
    Taken from the 6.S064 Machine Learning class at MIT 

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
def vecToImage(x, size = IMG_SIZE):
    """
    Taken from the 6.S064 Machine Learning class at MIT 
    """
    im = x/np.linalg.norm(x)
    im = im*(256./np.max(im))
    im.resize(*size)
    return im

def plotMeanByClass(X, y, showPlots=False):
    """
    Calculates the means of each digit class. showPlots=True 
    allows them to be plotted for the user (warning: blocking).
    """
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
        if showPlots:
            print "Now showing mean of %s" % str(uniques[i])
            showIm(means[i])

    # make into a matrix and return
    # each row is the character class [0-9]
    return np.vstack(means)

def kMeansFit(X, centers, metric):
    """
    Using the centers as our model and our distance metric, we
    fit X to our model, returning predictions for their labels

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
    X, Y = getTrainData()

    # split into training and testing
    (trainX, testX, trainY, testY) = sklearn.cross_validation.train_test_split(X, Y, test_size=TEST_RATIO)

    # try clustering by means
    init = None
    if meanInit:
        print "[*] Computing means of training dataset digits..."
        init = plotMeanByClass(trainX, trainY)
    else:
        # do random init
        print "[*] Computing random centers for initialization..."
        init = np.random.random_integers(0, high=256, size=(NUM_CLASSES, IMG_DIMENSION))
    
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
    del trainX, trainY, testX, testY

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

def kMeansWithPCA(meanInit=True, loadPickledData=False):
    """
    Tries to simplify the problem of clustering by projecting digit images
    into lower dimensional space with PCA and then clustering. Here we try a number of 
    values for l, the number of principal components

    meanInit        = do we initialize the centroids to means of the training set (else random)
    loadPickledData = do we load eigenvectors of covariance matrix of X (else calculate them)

    [*] RESULTS (with 2500 examples):

    accuracies = [0.1708, 0.2524,0.3684,0.4344,0.5332,0.5924,0.7204,0.7392,0.734,0.7492,0.7476,0.7476]
    l_values = [1, 5, 15, 50, 100, 200, 300, 400, 500, 600, 700, 784]
    """

    # try different values of number of components
    l_values = [1, 5, 15, 50, 100, 200, 300, 400, 500, 600, 700, 784]
    accs = []
    for l in l_values:

        X, Y = getTrainData()

        # get the eigenvectos of the covariance of X
        E = pca(X, loadFromPickle=True)

        # initialize clusters
        init = None
        if meanInit:
            print "[*] Computing means of training dataset digits..."
            init = plotMeanByClass(X, Y)
        else:
            # do random init
            print "[*] Computing random centers for initialization..."
            init = np.random.random_integers(0, high=256, size=(NUM_CLASSES, IMG_DIMENSION))

        print "[*] Projecting examples down into (%d) dimensional space..." % l
        projX = projection(X, E, l)
        projMeanX = projection(init, E, l)

        print "[*] Building clustering model..."
        centers, clusterAssignments = kMeans(projX, projMeanX, l2)

        # get training score
        trainingPredictions = kMeansFit(projX, centers, l2)
        accuracy = np.sum(trainingPredictions == Y) / float(Y.shape[0])
        print "[*] Training accuracy for %d components: %f" % (l, accuracy)

        accs.append(accuracy)

    # then plot the resulting curve to see the optimal values for PCA
    plt.figure('kMeans with PCA: Accuracy as a function of number of components')
    ax = plt.subplot(111)
    ax.plot(l_values, accs, label = "Accuracy")
    plt.xlabel('Number of components (l)')
    plt.ylabel('Accuracy')
    ax.legend(loc='lower right')
    plt.show()

def kNN(X, Y, z, dist, k):
    """
    PARAMS:

    X    = dataset                              (n x d)
    Y    = labels                               (n x 1)
    z    = example we are trying to classify    (1 x d)
    dist = distance metric
    k    = # of NN to consider (should be odd)
    """

    n = X.shape[0]

    bestK = [] # priority queue

    # search for k closest neighbors to z
    for i in range(n):
        distance = dist(X[i, :], z)
        heappush(bestK, (distance, i))
        bestK = nsmallest(k, bestK)

    # count the number of neighbors
    neighborsCount = {} # label => count
    for d, i in bestK:

        label = Y[i, 0]
        if not label in neighborsCount:
            neighborsCount[label] = 1
        else:
            neighborsCount[label] += 1

    # return the label for which the majority of the neighbors are
    # in ties, this simply chooses arbitrarily
    count, label = max([(c, l) for l, c in neighborsCount.iteritems()])
    return label

def testKNN():
    """
    Small test for kNN method, returns True upon sucess
    """
    X = np.matrix([[1,1], [2,1], [1,2], [1,3], [2,3]])
    Y = np.matrix([1, 1, 2, 2, 2]).T

    label1 = kNN(X, Y, np.matrix([1.5, 1.5]), l2, 3)
    label2 = kNN(X, Y, np.matrix([1.5, 1.5]), l2, 5)

    return label1 == 1 and label2 == 2

def kNNwithoutPCA(loadPickledData=True):
    """
    Simple application of kNN with varying values of k
    """

    print
    X, Y = getTrainData(loadPickledData)
    (trainX, testX, trainY, testY) = sklearn.cross_validation.train_test_split(X, Y, test_size=TEST_RATIO)
    numTestX = testX.shape[0]
    k_values = [1, 3, 5, 7, 9]
    accuracies = []

    print "[*] Beginning kNN testing without PCA..."
    for k in k_values:
        print "[*] Using k = %d ..." % k
        predictions = []
        actual = np.array(Y).flatten().tolist()
        for i in range(numTestX):
            example = testX[i, :]
            predictions.append(kNN(trainX, trainY, example, l2, k))

        accuracy = accuracy_score(actual, predictions)
        accuracies.append(accuracy)
        print "Accuracy for k = %d was %f" % (k, accuracy)

    # then plot the resulting curve to see the optimal values for k
    plt.figure('kNN without PCA: Accuracy as a function of k')
    ax = plt.subplot(111)
    ax.plot(k_values, accuracies, label = "Accuracy")
    plt.xlabel('Number nearest neighbors (k)')
    plt.ylabel('Accuracy')
    ax.legend(loc='lower right')
    #plt.show()
    plt.savefig("%s/kNN.png" % KNN_DIR)

def kNNPCA(loadPickledData=True):
    """
    Use PCA to reduce the dimensionality of the data through projection. 
    Then we explore varying number of principal components under fixed k. 
    """
    print
    X, Y = getTrainData(loadPickledData)
    (trainX, testX, trainY, testY) = sklearn.cross_validation.train_test_split(X, Y, test_size=TEST_RATIO)
    numTestX = testX.shape[0]
    k_values = [1, 3, 5, 7, 9]
    l_values = [1, 5, 15, 50, 100, 200, 300, 400, 500, 600, 784]
    accuracies = {}

    print "[*] Calculating eigenvectors of cov(X)..."
    E = pca(X, loadFromPickle=False)

    print "[*] Beginning kNN testing with PCA..."
    for k in k_values:
        
        # we are fixing k and seeing how varying l effects the results
        accuracies[k] = []

        for l in l_values:

            print "[*] Projecting X into l = %d dimensions, k = %d" % (l,k)
            projTrainX = projection(X, E, l)

            print "[*] Making predictions with kNN..."
            predictions = []
            actual = np.array(Y).flatten().tolist()
            for i in range(numTestX):
                example = projTrainX[i, :]
                predictions.append(kNN(projTrainX, trainY, example, l2, k))

            accuracy = accuracy_score(actual, predictions)
            accuracies[k].append(accuracy)
            print "Accuracy for l = %d, k = %d was %f" % (l, k, accuracy)

    for k in k_values:
        # then plot the resulting curve to see the optimal values for k
        plt.figure("kNN with PCA: Accuracy varying with l = %d and fixed k = %d" % (l, k))
        ax = plt.subplot(111)
        ax.plot(l_values, accuracies[k], label = "Accuracy")
        plt.xlabel('Number principal components (l)')
        plt.ylabel('Accuracy')
        ax.legend(loc='lower right')
        #plt.show()
        plt.savefig("%s/kNN-with-PCA-k-%d.png" % (KNN_DIR, k))

'''
############################ RESULTS ########################################
# Training accuracy: 0.103714
# Testing accuracy: 0.108794
digitRecognitionKMeans(meanInit=False, loadPickledData=True)

# Training accuracy: 0.762571
# Testing accuracy: 0.756635
# kaggle validation: 0.75
digitRecognitionKMeans(meanInit=True, loadPickledData=True)

#
kMeansWithPCA(meanInit=True, loadPickledData=True)

#
kMeansWithPCA(meanInit=False, loadPickledData=True)

# 
kNNwithoutPCA(loadPickledData=True)
'''

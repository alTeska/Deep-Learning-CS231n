import random
import numpy as np
import matplotlib.pyplot as plt
from dl4cv.features  import *
from dl4cv.data_utils import load_CIFAR10
from dl4cv.vis_utils import visualize_cifar10

from dl4cv.classifiers.linear_classifier import Softmax
from dl4cv.classifiers.neural_net import TwoLayerNet

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading extenrnal modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

def get_CIFAR10_data(num_training=48000, num_validation=1000, num_test=1000, num_dev=500):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for feature extraction and training.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets/'
    X, y = load_CIFAR10(cifar10_dir)


    # Our training set will be the first num_train points from the original
    # training set.
    mask = range(num_training)
    X_train = X[mask]
    y_train = y[mask]

    # Our validation set will be num_validation points from the original
    # training set.
    mask = range(num_training, num_training + num_validation)
    X_val = X[mask]
    y_val = y[mask]

    # We use a small subset of the training set as our test set.
    mask = range(num_training + num_validation, num_training + num_validation + num_test)
    X_test = X[mask]
    y_test = y[mask]

    return X, y, X_train, y_train, X_val, y_val, X_test, y_test

# Invoke the above function to get our data.
X_raw, y_raw, X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

num_color_bins = 10 # Number of bins in the color histogram
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

# Preprocessing: Subtract the mean feature
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

# Preprocessing: Divide by standard deviation. This ensures that each feature
# has roughly the same scale.
std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

# Preprocessing: Add a bias dimension
X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])


# Use the validation set to tune the learning rate and regularization strength

from dl4cv.classifiers.linear_classifier import Softmax
softmax = Softmax()

learning_rates = [1e-8, 1e-7]
regularization_strengths = [1e5, 1e6, 1e7]

results = {}
best_val = -1
best_softmax = None


################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the Softmax; save#
# the best trained classifer in best_softmax. You might also want to play      #
# with different numbers of bins in the color histogram. If you are careful    #
# you should be able to get accuracy of near 0.44 on the validation set.       #
################################################################################

for lr in learning_rates:
    for reg in regularization_strengths:
        softmax = Softmax()
        softmax.train(X_train_feats, y_train, learning_rate=lr, reg=reg, num_iters=100, verbose=False)
        y_train_pred = softmax.predict(X_train_feats)
        y_val_pred   = softmax.predict(X_val_feats)
        train_acc = np.mean(y_train == y_train_pred)
        valid_acc = np.mean(y_val   == y_val_pred)

        if valid_acc > best_val:
            best_val = valid_acc
            best_softmax = softmax
            best_set = [lr, reg]

        results[(lr, reg)] = [train_acc, valid_acc]

################################################################################
#                              END OF YOUR CODE                                #
################################################################################

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
          lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during validation: %f' % best_val)

# Evaluate your trained classifier on the test set
y_test_pred = best_softmax.predict(X_test_feats)
test_accuracy = np.mean(y_test == y_test_pred)
print(test_accuracy)

##NN
input_dim = X_train_feats.shape[1]
hidden_dim = 2500
num_classes = 10

net = TwoLayerNet(input_dim, hidden_dim, num_classes)
best_net = None

################################################################################
# TODO: Train a two-layer neural network on image features. You may want to    #
# validate various parameters as in previous sections. Store your best   #
# model in the best_net variable.                                              #
################################################################################

net = TwoLayerNet(input_dim, hidden_dim, num_classes)
# Train the network
stats = net.train(X_train_feats, y_train, X_val_feats, y_val,
num_iters=7000, batch_size=256,
learning_rate=0.9, learning_rate_decay=0.98,
reg=1e-3, verbose=True)

# Predict on the validation set
val_acc = (net.predict(X_val_feats) == y_val).mean()
print('Validation accuracy: ', val_acc)
best_net = net
'''
hidden_dim = [600, 700, 800, 900, 1000]
learning_rates = [1e-3, 1e-4, 1e-5]
regularization_strengths = [0.1, 0.3, 0.5,0.7, 0.9, 1]


# Train the network
for hd in hidden_dim:
    net = TwoLayerNet(input_dim, hd, num_classes)
    for lr in learning_rates:
        for rg in regularization_strengths:

            ## Train the network
            stats = net.train(X_train_feats, y_train, X_val_feats, y_val,
            num_iters=2000, batch_size=256,
            learning_rate=lr, learning_rate_decay=0.96,
            reg=rg, verbose=False)

            val_acc = (net.predict(X_val_feats) == y_val).mean()
            if val_acc > best_val:
                best_val = val_acc
                best_net = net
                best_set = [lr, rg, hd]
            print(rg, lr, hd, val_acc)

# Predict on the validation set
print('Validation accuracy: ', best_val)
net = TwoLayerNet(input_dim, 10000, num_classes)
best_stats = net.train(X_train_feats, y_train, X_val_feats, y_val,
        num_iters=10000, batch_size=256,
        learning_rate=best_set[0], learning_rate_decay=0.96,
        reg=best_set[1], verbose=False)

################################################################################
#                              END OF YOUR CODE                                #
################################################################################

# Run your neural net classifier on the test set. You should be able to
# get more than 55% accuracy.
test_acc = (net.predict(X_test_feats) == y_test).mean()
val_acc = (net.predict(X_val_feats) == y_val).mean()
print("Test accuracy: ", test_acc)
print('Validation accuracy: ', val_acc)
'''

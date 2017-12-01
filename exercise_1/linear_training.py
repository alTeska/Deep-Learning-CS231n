import numpy as np
from dl4cv.data_utils import load_CIFAR10
from dl4cv.classifiers import Softmax
#from dl4cv.classifiers.softmax import *
softmax = Softmax()

# Load the raw CIFAR-10 data
cifar10_dir = 'datasets/'
X, y = load_CIFAR10(cifar10_dir)

# Split the data into train, val, and test sets. In addition we will
# create a small development set as a subset of the data set;
# we can use this for development so our code runs faster.
num_training = 48000
num_validation = 1000
num_test = 1000
num_dev = 500

assert (num_training + num_validation + num_test) == 50000, 'You have not provided a valid data split.'

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

# We will also make a development set, which is a small subset of
# the training set. This way the development cycle is faster.
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
X_dev -= mean_image

X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

loss_hist = softmax.train(X_train, y_train, learning_rate=1e-7, reg=5e4, num_iters=10, verbose=True)

y_train_pred = softmax.predict(X_train)
print('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
y_val_pred = softmax.predict(X_val)
print('validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))

'''
#TODO:
# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results = {}
best_val = -1
best_softmax = None
all_classifiers = []

learning_rates = [ 3e-7, 5e-7, 8e-7, 9e-7]
#learning_rates = [3e-7, 5e-7, 8e-7]
regularization_strengths = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]

################################################################################
# TODO:                                                                        #
# Write code that chooses the best hyperparameters by tuning on the validation #
# set. For each combination of hyperparameters, train a classifier on the      #
# training set, compute its accuracy on the training and validation sets, and  #
# store these numbers in the results dictionary. In addition, store the best   #
# validation accuracy in best_val and the Softmax object that achieves this    #
# accuracy in best_softmax.                                                    #
#                                                                              #
# Hint: You should use a small value for num_iters as you develop your         #
# validation code so that the classifiers don't take much time to train;       #
# once you are confident that your validation code works, you should rerun     #
# the validation code with a larger value for num_iters.                       #
################################################################################

# INP: (learning_rate    , regularization_strength)
# OUT: (training_accuracy, validation_accuracy    )
print('lr, reg, valid_acc')
for lr in learning_rates:
    for reg in regularization_strengths:

        softmax = Softmax()
        softmax.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=1000 ,verbose=False)
        all_classifiers.append(softmax)

        y_train_pred = softmax.predict(X_train)
        y_val_pred   = softmax.predict(X_val)
        train_acc = np.mean(y_train == y_train_pred)
        valid_acc = np.mean(y_val   == y_val_pred)
        print(lr, reg, valid_acc)

        if valid_acc > best_val:
            best_val = valid_acc
            best_softmax = softmax
            best_set = [lr, reg]

        results[(lr, reg)] = [train_acc, valid_acc]

softmax = Softmax()
softmax.train(X_train, y_train, learning_rate=best_set[0], reg=best_set[1], num_iters=5000, verbose=True)
best_softmax = softmax

y_train_pred = softmax.predict(X_train)
y_val_pred   = softmax.predict(X_val)
train_acc = np.mean(y_train == y_train_pred)
valid_acc = np.mean(y_val   == y_val_pred)

print('train_acc %f' % train_acc)
print('valid_acc %f' % valid_acc)
################################################################################
#                              END OF YOUR CODE                                #
################################################################################
#sorted_classifiers = sorted(all_classifiers, key=lambda x : x[1])
#best_softmax = sorted_classifiers[-1][0]

# Print out results.
for (lr, reg) in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
          lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during validation: %f' % best_val)

y_test_pred = best_softmax.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))
'''

from dl4cv.data_utils import load_CIFAR10
from dl4cv.vis_utils import visualize_cifar10
from dl4cv.classifiers.neural_net import TwoLayerNet
import numpy as np


def get_CIFAR10_data(num_training=48000, num_validation=1000, num_test=1000, num_dev=500):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier.
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

    # We will also make a development set, which is a small subset of
    # the training set. This way the development cycle is faster.
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    return X, y, X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev


# Invoke the above function to get our data.
X_raw, y_raw, X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev= get_CIFAR10_data()

#input_size = 32 * 32 * 3
#hidden_size = 50
#num_classes = 10
#net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
#stats = net.train(X_train, y_train, X_val, y_val,
#            num_iters=5000, batch_size=200,
#            learning_rate=1e-4, learning_rate_decay=0.9,
#            reg=0.5, verbose=True)
#
## Predict on the validation set
#val_acc = (net.predict(X_val) == y_val).mean()
#print('Validation accuracy: ', val_acc)

best_val = -1

input_size = 32 * 32 * 3
num_classes = 10
hidden_size = [50, 100, 150]
learning_rates = [1e-3, 1e-4]
regularization_strengths = [1, 5, 10]
learning_rate_decays = 0.95


################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
best_val = -1

input_size = 32 * 32 * 3
num_classes = 10
hidden_size = [100, 125, 150, 200]
learning_rates = [1e-3, 1e-4]
regularization_strengths = [0.8, 0.9, 1, 1.1]
learning_rate_decays = 0.95


print('hs', 'rg', 'lr', 'Val acc')

for hs in hidden_size:
    for lr in learning_rates:
        for rg in regularization_strengths:
            # Train the network
            net = TwoLayerNet(input_size, hs, num_classes)
            stats = net.train(X_train, y_train, X_val, y_val,
                    num_iters=2000, batch_size=200,
                    learning_rate=lr, learning_rate_decay=0.98,
                    reg=rg, verbose=False)

            # Predict on the validation set
            val_acc = (net.predict(X_val) == y_val).mean()
            if val_acc > best_val:
                best_val = val_acc
                best_net = net
                best_set = [lr, rg, hs]

            print(hs, rg, lr, val_acc)

print(best_val, best_set)
best_net = TwoLayerNet(input_size, best_set[2], num_classes)
best_stats = best_net.train(X_train, y_train, X_val, y_val,
        num_iters=10000, batch_size=200,
        learning_rate=best_set[0], learning_rate_decay=0.98,
        reg=best_set[1], verbose=False)

val_acc = (best_net.predict(X_val) == y_val).mean()
print(val_acc)

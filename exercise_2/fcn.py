import time
import numpy as np
import matplotlib.pyplot as plt
from dl4cv.classifiers.fc_net import *
from dl4cv.data_utils import get_CIFAR10_data
from dl4cv.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from dl4cv.solver import Solver

import warnings
warnings.filterwarnings('ignore')

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Load the (preprocessed) CIFAR10 data.
data = get_CIFAR10_data()

N, D, H1, H2, C = 2, 15, 20, 30, 10
X = np.random.randn(N, D)
y = np.random.randint(C, size=(N,))

model = FullyConnectedNet([H1,H2,H2], input_dim=D, num_classes=C, weight_scale=5e-2, dtype=np.float64)
loss, grads = model.loss(X, y)

'''
print('Initial loss: ', loss)
for reg in [0, 3.14]:
  print('Running check with reg = ', reg)
  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                            reg=reg, weight_scale=5e-2, dtype=np.float64)

  loss, grads = model.loss(X, y)
  print('Initial loss: ', loss)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
    print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
'''


best_model = None
best_val = -1
################################################################################
# TODO: Train the best FullyConnectedNet that you can on CIFAR-10. You might   #
# batch normalization and dropout useful. Store your best model in the         #
# best_model variable.                                                         #
# Note that dropout is not required to pass beyond the linear scoring regime   #
################################################################################
from dl4cv.features import *

X_raw = data['X_raw']
y_raw = data['y_raw']
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']
'''
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

data['X_train'] = X_train_feats
data['X_val']   = X_val_feats
data['X_test']  = X_test_feats
'''
best_model = None
best_val = -1

hidden_dims = [120, 105, 90, 75]
upadate_rules = ['adam']
learning_rates = [1e-3]
num_epochs = [10,20]
l_decays = [0.95]
regularization_strengths = [1e-4, 1e-3, 1e-2]
dropout_choices = [0.0, 0.25, 0.5, 0.75] #0.25

print('ur', 'lr', 'ne', 'ld','rs','dp', 'valAcc', 'valTest')

for ur in upadate_rules:
    for lr in learning_rates:
        for ne in num_epochs:
            for ld in l_decays:
                for rs in regularization_strengths:
                    for dp in dropout_choices:

                        model = FullyConnectedNet(hidden_dims, dropout=dp, reg=rs, weight_scale=5e-2, use_batchnorm=True)
                        solver = Solver(model, data,
                        lr_decay=ld,
                        num_epochs=ne,
                        batch_size=100,
                        update_rule=ur,
                        optim_config={
                        'learning_rate': lr
                        },
                        verbose=False)
                        solver.train()

                        y_val_pred = np.argmax(model.loss(X_val), axis=1)
                        y_test_pred = np.argmax(model.loss(X_test), axis=1)
                        val_acc = (y_val_pred == y_val).mean()
                        val_test = (y_test_pred == y_test).mean()

                        if val_acc > best_val:
                            best_model = model
                            best_val = val_acc

                        print(ur, lr, ne, ld, rs, dp, val_acc, val_test)


################################################################################
#                              END OF YOUR CODE                                #
################################################################################


################################################################################
#                              END OF YOUR CODE                                #
################################################################################

X_test = data['X_test']
X_val = data['X_val']
y_val = data['y_val']
y_test = data['y_test']

y_test_pred = np.argmax(best_model.loss(X_test), axis=1)
y_val_pred = np.argmax(best_model.loss(X_val), axis=1)
print('Validation set accuracy: ', (y_val_pred == y_val).mean())
print('Test set accuracy: ', (y_test_pred == y_test).mean())

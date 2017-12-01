import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # W.shape (3073, 10)
    # X.shape (500, 3073)
    # y.shape (500,)
    # D = 3073
    # C = 10
    # N = 500
    #
    # TODO: Compute the softmax loss and its gradient using explicit loops.
    # Store the loss in loss and the gradient in dW. If you are not careful
    # here, it is easy to run into numeric instability. Don't forget the
    # regularization!                                                           #
    ##########################################################################

    D = W.shape[0]
    C = W.shape[1]
    N = X.shape[0]

    for i in np.arange(N):
        #loss
        scores = X[i,:].transpose().dot(W)
        scores -= np.max(scores)

        sum_exp = 0
        for s in scores:
            sum_exp += np.exp(s)
        val = np.exp(scores[y[i]]) / sum_exp
        loss += -np.log(val)

        #gradient
        for k in np.arange(C):
            p_k = np.exp(scores[k]) / sum_exp
            dW[:, k] += (p_k - (k == y[i])) * X[i]

    #regularization
    dW /= N
    dW += reg * W

    loss /= N
    loss += 0.5 * reg * np.sum(W * W)

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N = X.shape[0]
    (C, D) = W.shape
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.
    # Store the loss in loss and the gradient in dW. If you are not careful
    # here, it is easy to run into numeric instability. Don't forget the
    # regularization!                                                           #
    #############################################################################

    #loss
    num_train = X.shape[0]
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)
    sum_scores = np.sum(np.exp(scores), axis=1, keepdims=True)

    p    = np.exp(scores)/sum_scores
    loss = np.sum(-np.log(p[np.arange(num_train), y]))

    #gradient
    dS = p.copy()
    dW = (X.T).dot(dS)
    dS[range(num_train), list(y)] += -1

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

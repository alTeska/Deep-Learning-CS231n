from dl4cv.layers import *

def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def affine_batch_relu_forward(x, w, b, gamma, beta, bn_param):
    a, fc_cache = affine_forward(x, w, b)
    b, b_cache  = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(b)

    cache = (fc_cache, b_cache, relu_cache)
    return out, cache


def affine_batch_relu_backward(dout, cache):
    fc_cache, b_cache, relu_cache = cache

    da = relu_backward(dout, relu_cache)
    dbatch, dbeta, dgamma = batchnorm_backward(da, b_cache)
    dx, dw, db = affine_backward(dbatch, fc_cache)

    return dx, dw, db, dgamma, dbeta


def affine_batch_relu_dpout_forward(x, w, b, gamma, beta, bn_param, dropout_param):
    a, fc_cache = affine_forward(x, w, b)
    b, b_cache  = batchnorm_forward(a, gamma, beta, bn_param)
    r, relu_cache = relu_forward(b)
    out, dp_cache = dropout_forward(r, dropout_param)

    cache = (fc_cache, b_cache, relu_cache, dp_cache)
    return out, cache


def affine_batch_relu_dpout_backward(dout, cache):
    fc_cache, b_cache, relu_cache, dp_cache = cache

    ddrop_out = dropout_backward(dout, dp_cache)
    da = relu_backward(ddrop_out, relu_cache)
    dbatch, dbeta, dgamma = batchnorm_backward(da, b_cache)
    dx, dw, db = affine_backward(dbatch, fc_cache)

    return dx, dw, db, dgamma, dbeta

def affine_relu_dpout_forward(x, w, b, dropout_param):
    a, fc_cache = affine_forward(x, w, b)
    r, relu_cache = relu_forward(a)
    out, dp_cache = dropout_forward(r, dropout_param)

    cache = (fc_cache, relu_cache, dp_cache)
    return out, cache


def affine_relu_dpout_backward(dout, cache):
    fc_cache, relu_cache, dp_cache = cache

    ddrop_out = dropout_backward(dout, dp_cache)
    da = relu_backward(ddrop_out, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)

    return dx, dw, db

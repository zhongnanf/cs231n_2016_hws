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
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    num_samp = X.shape[0]
    num_class = W.shape[1]

    sum_Li = 0

    for ii in xrange(num_samp):
        score = X[ii].dot(W)
        score -= np.max(score)
        score = np.exp(score)
        correct_class_score = score[y[ii]]
        sum_Li += -np.log(correct_class_score /np.sum(score))

    loss = sum_Li / num_samp + 0.5 * reg * np.sum(W * W)

    score = np.exp(X.dot(W))
    score = score / np.reshape(np.sum(score,axis=1),[-1, 1])

    dW = np.zeros(W.shape)
    for kk in xrange(num_class):
        dWk = np.zeros([1,X.shape[1]])
        for ii in xrange(num_samp):
            if y[ii] != kk:
                dWk += score[ii,kk]*X[ii]
            else:
                dWk += (-1+score[ii,kk])*X[ii]
        dW[:,kk] = dWk

    dW = dW/num_samp+reg*W

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_samp = X.shape[0]
    num_class = W.shape[1]

    score = np.exp(np.dot(X,W))
    f = np.sum(score, axis=1)
    loss = np.sum(-np.log(score[range(num_samp),y] / f))
    loss = loss/num_samp + 0.5*reg*np.sum(W*W)

    p = score / np.reshape(f,[-1,1])
    for kk in xrange(num_class):
        ind = (y==kk).nonzero()[0]
        p[ind,kk] += -1

    dW = np.dot(np.transpose(X),p)/num_samp + reg*W



    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class convNet(object):
    """
    A multi-layer convolutional network with the following architecture:

    (conv 5 - relu - 2x2 max pool) x P -
    (affine - relu) x Q -
    affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32),
                 num_convLys=2, filter_size=(5, 5), num_filters=(6, 16), use_sBatchnorm=True,
                 num_afLys=2, hidden_dims=(120, 84), use_batchnorm=True,
                 num_classes=10,
                 weight_scale=1e-3, reg=0.0, dtype=np.float32):

        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.num_convLys = num_convLys
        self.num_afLys = num_afLys
        self.use_sBatchnorm = use_sBatchnorm
        self.use_batchnorm = use_batchnorm
        self.filter_size = filter_size

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################

        # Conv layers
        C, H, W = input_dim

        HH = H
        WW = W
        for cnt1 in xrange(num_convLys):

            if cnt1 == 0:
                ch = C
            else:
                ch = num_filters[cnt1 - 1]

            self.params['CW%d' % (cnt1 + 1)] = np.random.normal(scale=weight_scale,
                                                                size=[num_filters[cnt1], ch, filter_size[cnt1],
                                                                      filter_size[cnt1]])
            self.params['Cb%d' % (cnt1 + 1)] = np.zeros([num_filters[cnt1], ])
            HH //= 2
            WW //= 2

            if HH <= 0:
                raise ValueError('HH cannot be less than 1')
            if WW <= 0:
                raise ValueError('WW cannot be less than 1')

        if self.use_sBatchnorm:
            for cnt1 in xrange(num_convLys):
                self.params['Cgamma%d' % (cnt1 + 1)] = np.ones((num_filters[cnt1],))
                self.params['Cbeta%d' % (cnt1 + 1)] = np.zeros((num_filters[cnt1],))

        # Affine layers
        self.params['AW1'] = np.random.normal(scale=weight_scale, size=(num_filters[-1] * HH * WW, hidden_dims[0]))
        self.params['Ab1'] = np.zeros([hidden_dims[0], ])

        for cnt1 in xrange(2, num_afLys + 1):
            self.params['AW%d' % cnt1] = np.random.normal(scale=weight_scale,
                                                          size=(hidden_dims[cnt1 - 2], hidden_dims[cnt1 - 1]))
            self.params['Ab%d' % cnt1] = np.zeros([hidden_dims[cnt1 - 1], ])

        # Last affine layer

        self.params['AW%d' % (self.num_afLys + 1)] = np.random.normal(scale=weight_scale,
                                                                      size=(hidden_dims[-1], num_classes))
        self.params['Ab%d' % (self.num_afLys + 1)] = np.zeros([num_classes, ])

        if self.use_batchnorm:
            for cnt1 in xrange(len(hidden_dims)):
                self.params['Agamma%d' % (cnt1 + 1)] = np.ones((hidden_dims[cnt1],))
                self.params['Abeta%d' % (cnt1 + 1)] = np.zeros((hidden_dims[cnt1],))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        """

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################

        conv_param = []
        for cnt1 in xrange(len(self.filter_size)):
            conv_param.append({'stride': 1, 'pad': (self.filter_size[cnt1] - 1) / 2})

        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        sbn_params = []
        if self.use_sBatchnorm:
            sbn_params = [{'mode': 'train'} for i in xrange(self.num_convLys)]

        c_conv_cache = []
        c_relu_cache = []
        c_max_pool_cache = []
        if self.use_sBatchnorm:
            c_sbn_cache = []

        # Conv Layers
        for cnt1 in xrange(self.num_convLys):
            W = self.params['CW%d' % (cnt1 + 1)]
            b = self.params['Cb%d' % (cnt1 + 1)]

            X, cache = conv_forward_fast(X, W, b, conv_param[cnt1])
            c_conv_cache.append(cache)
            X, cache = relu_forward(X)
            c_relu_cache.append(cache)
            X, cache = max_pool_forward_fast(X, pool_param)
            c_max_pool_cache.append(cache)

            if self.use_sBatchnorm:
                gamma = self.params['Cgamma%d' % (cnt1 + 1)]
                beta = self.params['Cbeta%d' % (cnt1 + 1)]
                X, cache = spatial_batchnorm_forward(X, gamma, beta, sbn_params[cnt1])
                c_sbn_cache.append(cache)

        # Affine Layers

        bn_params = []
        if self.use_batchnorm:
            bn_params = [{'mode': 'train'} for i in xrange(self.num_afLys)]

        a_affine_cache = []
        a_relu_cache = []
        if self.use_batchnorm:
            a_bn_cache = []

        X_convAf_shape = X.shape

        X = X.reshape([X.shape[0], -1])

        for cnt1 in xrange(self.num_afLys):
            W = self.params['AW%d' % (cnt1 + 1)]
            b = self.params['Ab%d' % (cnt1 + 1)]
            X, cache = affine_forward(X, W, b)
            a_affine_cache.append(cache)

            X, cache = relu_forward(X)
            a_relu_cache.append(cache)

            if self.use_batchnorm:
                gamma = self.params['Agamma%d' % (cnt1 + 1)]
                beta = self.params['Abeta%d' % (cnt1 + 1)]
                X, cache = batchnorm_forward(X, gamma, beta, bn_params[cnt1])
                a_bn_cache.append(cache)

        # Last Affine Layer

        W = self.params['AW%d' % (self.num_afLys + 1)]
        b = self.params['Ab%d' % (self.num_afLys + 1)]
        scores, cache = affine_forward(X, W, b)
        a_affine_cache.append(cache)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dX = softmax_loss(scores, y)

        # calc loss with regularizations

        for cnt1 in xrange(self.num_convLys):
            loss += 0.5 * self.reg * np.sum(self.params['CW%d' % (cnt1 + 1)] * self.params['CW%d' % (cnt1 + 1)])

        for cnt1 in xrange(self.num_afLys + 1):
            loss += 0.5 * self.reg * np.sum(self.params['AW%d' % (cnt1 + 1)] * self.params['AW%d' % (cnt1 + 1)])

        # Last Affine Layer
        dX, dW, db = affine_backward(dX, a_affine_cache[-1])
        grads['AW%d' % (self.num_afLys + 1)] = dW
        grads['Ab%d' % (self.num_afLys + 1)] = db

        # Affine Layers
        for cnt1 in xrange(self.num_afLys):
            if self.use_batchnorm:
                dX, dgamma, dbeta = batchnorm_backward_alt(dX, a_bn_cache[self.num_afLys - cnt1 - 1])
                grads['Agamma%d' % (self.num_afLys - cnt1)] = dgamma
                grads['Abeta%d' % (self.num_afLys - cnt1)] = dbeta

            dX = relu_backward(dX,a_relu_cache[self.num_afLys - cnt1 - 1])

            dX, dW, db = affine_backward(dX, a_affine_cache[self.num_afLys - cnt1 - 1])
            grads['AW%d' % (self.num_afLys - cnt1)] = dW
            grads['Ab%d' % (self.num_afLys - cnt1)] = db

        dX = dX.reshape(X_convAf_shape)

        # Conv Layers
        for cnt1 in xrange(self.num_convLys):
            if self.use_sBatchnorm:
                dX, dgamma, dbeta = spatial_batchnorm_backward(dX, c_sbn_cache[self.num_convLys - cnt1 - 1])
                grads['Cgamma%d' % (self.num_convLys - cnt1)] = dgamma
                grads['Cbeta%d' % (self.num_convLys - cnt1)] = dbeta

            dX = max_pool_backward_fast(dX, c_max_pool_cache[self.num_convLys - cnt1 - 1])

            dX = relu_backward(dX, c_relu_cache[self.num_convLys - cnt1 - 1])

            dX, dW, db = conv_backward_fast(dX, c_conv_cache[self.num_convLys - cnt1 - 1])
            grads['CW%d' % (self.num_convLys - cnt1)] = dW
            grads['Cb%d' % (self.num_convLys - cnt1)] = db

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


pass

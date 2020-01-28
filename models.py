from modules import *
import numpy as np

class TwoLayerNet:
    # fc - relu - fc - softmax

    def __init__(self, input_dim, hidden_dim, num_classes, weight_scale=1e-3, reg=0.0):
        self.params = {}
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        self.reg = reg

    def loss(self, X, y=None):
        hidden_layer, hid_cache = forward(X, self.params['W1'], self.params['b1'])
        hidden_layer_relu, hid_relu_cache = relu_forward(hidden_layer)
        scores, scores_cache = forward(hidden_layer_relu, self.params['W2'], self.params['b2'])

        if y is None:
            return scores

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(np.square(self.params['W1']))
        loss += 0.5 * self.reg * np.sum(np.square(self.params['W2']))

        grads = {}
        dhidden_layer_relu, grads['W2'], grads['b2'] = backward(dscores, scores_cache)
        dhidden_layer = relu_backward(dhidden_layer_relu, hid_relu_cache)
        dX, grads['W1'], grads['b1'] = backward(dhidden_layer, hid_cache)
        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']

        return loss, grads


class FullyConnectedNet:


    def __init__(self, input_dim, hidden_dims, num_classes,
                 weight_scale=1e-2, reg=0.0,
                 normalization=None, dropout=1, seed=None):

        dims = [input_dim] + hidden_dims + [num_classes]
        self.num_layers = len(hidden_dims) + 1
        self.reg = reg
        self.normalization = normalization
        self.use_dropout = dropout != 1

        self.params = {}
        for i in range(1, self.num_layers+1):
            self.params['W' + str(i)] = weight_scale * np.random.randn(dims[i-1], dims[i])
            self.params['b' + str(i)] = np.zeros(dims[i])
            if (self.normalization == 'batchnorm' or self.normalization == 'layernorm') and i != self.num_layers:
                self.params['gamma' + str(i)] = np.ones(dims[i])
                self.params['beta' + str(i)] = np.zeros(dims[i])

        self.bn_params = []
        if self.normalization == 'batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers-1)]

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None: self.dropout_param['seed'] = seed


    def loss(self, X, y=None):
        mode = 'test' if y is None else 'train'
        if self.normalization == 'batchnorm':
            for bn_param in self.bn_params: bn_param['mode'] = mode
        if self.use_dropout:
            self.dropout_param['mode'] = mode

        caches = []
        layer = X
        for i in range(1, self.num_layers+1):
            layer, layer_cache = forward(layer, self.params['W' + str(i)], self.params['b' + str(i)])
            caches.append(layer_cache)
            if i != self.num_layers:
                if self.normalization == 'batchnorm':
                    layer, batchnorm_cache = batchnorm_forward(layer, self.params['gamma' + str(i)],
                                                               self.params['beta' + str(i)], self.bn_params[i-1])
                    caches.append(batchnorm_cache)
                elif self.normalization == 'layernorm':
                    layer, layernorm_cache = layernorm_forward(layer, self.params['gamma' + str(i)], self.params['beta' + str(i)])
                    caches.append(layernorm_cache)
                layer, relu_cache = relu_forward(layer)
                caches.append(relu_cache)
                if self.use_dropout:
                    layer, dropout_cache = dropout_forward(layer, self.dropout_param)
                    caches.append(dropout_cache)

        if mode == 'test':
            return layer

        loss, dlayer = softmax_loss(layer, y)
        grads = {}
        for i in reversed(range(1, self.num_layers+1)):
            loss += 0.5 * self.reg * np.sum(np.square(self.params['W' + str(i)]))

            if i != self.num_layers:
                if self.use_dropout:
                    dlayer = dropout_backward(dlayer, caches.pop())
                dlayer = relu_backward(dlayer, caches.pop())
                if self.normalization == 'batchnorm':
                    dlayer, grads['gamma' + str(i)], grads['beta' + str(i)] = batchnorm_backward_alt(dlayer, caches.pop())
                elif self.normalization == 'layernorm':
                    dlayer, grads['gamma' + str(i)], grads['beta' + str(i)] = layernorm_backward(dlayer, caches.pop())
            dlayer, grads['W' + str(i)], grads['b' + str(i)] = backward(dlayer, caches.pop())
            grads['W' + str(i)] += self.reg * self.params['W' + str(i)]

        return loss, grads


class ThreeLayerConvNet:
    # conv - relu - 2x2 max pool - flatten - fc - relu - fc - softmax

    def __init__(self, input_dim=(3, 32, 32),
                 num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):

        self.params = {}
        self.params['W1'] = weight_scale * np.random.randn(num_filters, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale * np.random.randn(num_filters * (input_dim[1]//2) * (input_dim[2]//2), hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        self.reg = reg

    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2} # padding and stride chosen to preserve the input spatial size
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2} # 2x2 max pooling with stride 2 halves the input height and width

        conv_layer, conv_cache = conv_forward(X, self.params['W1'], self.params['b1'], conv_param) # conv
        conv_layer_relu, conv_relu_cache = relu_forward(conv_layer) # relu
        max_pool_layer, max_pool_cache = max_pool_forward(conv_layer_relu, pool_param) # 2x2 max pool
        flatten_layer, flatten_cache = flatten_forward(max_pool_layer) # flatten
        hidden_layer, hidden_cache = forward(flatten_layer, self.params['W2'], self.params['b2']) # fc
        hidden_layer_relu, hidden_relu_cache = relu_forward(hidden_layer) # relu
        scores, scores_cache = forward(hidden_layer_relu, self.params['W3'], self.params['b3']) # fc

        if y is None:
            return scores

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(np.square(self.params['W1']))
        loss += 0.5 * self.reg * np.sum(np.square(self.params['W2']))
        loss += 0.5 * self.reg * np.sum(np.square(self.params['W3']))

        grads = {}
        dhidden_layer_relu, grads['W3'], grads['b3'] = backward(dscores, scores_cache)
        dhidden_layer = relu_backward(dhidden_layer_relu, hidden_relu_cache)
        dflatten_layer, grads['W2'], grads['b2'] = backward(dhidden_layer, hidden_cache)
        dmax_pool_layer = flatten_backward(dflatten_layer, flatten_cache)
        dconv_relu_layer = max_pool_backward(dmax_pool_layer, max_pool_cache)
        dconv_layer = relu_backward(dconv_relu_layer, conv_relu_cache)
        dX, grads['W1'], grads['b1'] = conv_backward(dconv_layer, conv_cache)
        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']
        grads['W3'] += self.reg * self.params['W3']

        return loss, grads

import numpy as np


''' Layers '''

def forward(x, w, b):
    # x: [N x D], w: [D x M], b: [M]
    out = x.dot(w) + b
    cache = (x, w, b)
    return out, cache

def backward(dout, cache):
    # x: [N x D], w: [D x M], b: [M], dout: [N x M]
    x, w, b = cache
    db = np.sum(dout, axis=0)
    dw = x.T.dot(dout)
    dx = dout.dot(w.T)
    return dx, dw, db

def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    x = cache
    dx = (x > 0) * dout
    return dx

def flatten_forward(x):
    out = x.reshape((x.shape[0], -1))
    cache = x.shape
    return out, cache

def flatten_backward(dout, cache):
    shape = cache
    dx = dout.reshape(shape)
    return dx

def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    running_mean = bn_param.get('running_mean', np.zeros(x.shape[1]))
    running_var = bn_param.get('running_var', np.zeros(x.shape[1]))
    cache = None

    if mode == 'train':
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        x_norm = (x - batch_mean) / np.sqrt(batch_var + eps)
        out = gamma * x_norm + beta

        running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        running_var = momentum * running_var + (1 - momentum) * batch_var
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var

        cache = (x, batch_mean, batch_var, x_norm, gamma, eps)
    elif mode == 'test':
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta

    return out, cache

def batchnorm_backward(dout, cache):
    x, batch_mean, batch_var, x_norm, gamma, eps = cache
    N = x.shape[0]

    dgamma = np.sum(x_norm * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx_norm = dout * gamma

    dx = (1 / np.sqrt(batch_var + eps)) * dx_norm
    dbatch_var = ((x - batch_mean) * (-1/2) * (batch_var + eps)**(-3/2) * dx_norm).sum(axis=0)
    dbatch_mean = (-1 / np.sqrt(batch_var + eps)) * dx_norm.sum(axis=0)
    dbatch_mean += (-2/N) * (x - batch_mean).sum(axis=0) * dbatch_var
    dx += (2/N) * (x - batch_mean) * dbatch_var
    dx += np.ones_like(x) * (1/N) * dbatch_mean

    return dx, dgamma, dbeta

def batchnorm_backward_alt(dout, cache):
    # galima išsivesti sutrauktą formulę ant popieriaus; šiek tiek pagreitina veikimą
    x, batch_mean, batch_var, x_norm, gamma, eps = cache
    N = x.shape[0]

    dgamma = np.sum(dout*x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)

    dx = (1/N) * gamma * (batch_var + eps)**(-1/2) * (
            dout * N
            - (x - batch_mean) * (dout*(x - batch_mean)).sum(axis=0) / (batch_var + eps)
            - dout.sum(axis=0)
         )

    return dx, dgamma, dbeta

def layernorm_forward(x, gamma, beta):
    eps = 1e-5

    layer_mean = np.mean(x, axis=1, keepdims=True)
    layer_var = np.var(x, axis=1, keepdims=True)
    x_norm = (x - layer_mean) / np.sqrt(layer_var + eps)
    out = gamma * x_norm + beta

    cache = (x, layer_mean, layer_var, x_norm, gamma, eps)
    return out, cache

def layernorm_backward(dout, cache):
    x, layer_mean, layer_var, x_norm, gamma, eps = cache
    D = x.shape[1]

    dgamma = np.sum(x_norm * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx_norm = dout * gamma

    dx = (1 / np.sqrt(layer_var + eps)) * dx_norm
    dlayer_var = ((x - layer_mean) * (-1/2) * (layer_var + eps)**(-3/2) * dx_norm).sum(axis=1, keepdims=True)
    dlayer_mean = (-1 / np.sqrt(layer_var + eps)) * dx_norm.sum(axis=1, keepdims=True)
    dlayer_mean += (-2/D) * (x - layer_mean).sum(axis=1, keepdims=True) * dlayer_var
    dx += (2/D) * (x - layer_mean) * dlayer_var
    dx += np.ones_like(x) * (1/D) * dlayer_mean

    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    mode = dropout_param['mode']
    p = dropout_param['p']
    if 'seed' in dropout_param: np.random.seed(dropout_param['seed'])
    cache = None

    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        cache = mask
    if mode == 'test':
        out = x

    return out, cache

def dropout_backward(dout, cache):
    mask = cache
    dx = mask * dout
    return dx

def conv_forward(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)

    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    x_pad = np.pad(x, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    out = np.zeros((N, F, H_out, W_out))

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    x_slice = x_pad[n, :, (i*stride):(i*stride + HH), (j*stride):(j*stride + WW)]
                    out[n, f, i, j] = np.sum(x_slice * w[f]) + b[f]

    cache = (x, w, b, conv_param)
    return out, cache

def conv_backward(dout, cache):
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 1)

    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    x_pad = np.pad(x, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    dx_pad[n, :, (i*stride):(i*stride + HH), (j*stride):(j*stride + WW)] += w[f] * dout[n, f, i, j]
                    dw[f] += x_pad[n, :, (i*stride):(i*stride + HH), (j*stride):(j*stride + WW)] * dout[n, f, i, j]
                    db[f] += dout[n, f, i, j]

    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    return dx, dw, db

def max_pool_forward(x, pool_param):
    N, C, H, W = x.shape
    pool_height = pool_param.get('pool_height', 1)
    pool_width = pool_param.get('pool_width', 1)
    stride = pool_param.get('stride', 1)

    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    out = np.zeros((N, C, H_out, W_out))

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    x_slice = x[n, c, (i*stride):(i*stride + pool_height), (j*stride):(j*stride + pool_width)]
                    out[n, c, i, j] = np.max(x_slice)

    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param.get('pool_height', 2)
    pool_width = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)

    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    x_slice = x[n, c, (i*stride):(i*stride + pool_height), (j*stride):(j*stride + pool_width)]
                    max_index = np.unravel_index(x_slice.argmax(), x_slice.shape)
                    dx[n, c, (i*stride):(i*stride+pool_height), (j*stride):(j*stride+pool_width)][max_index] = dout[n, c, i, j]

    return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    N, C, H, W = x.shape
    x = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    N, C, H, W = dout.shape
    dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    eps = gn_param.get('eps',1e-5)

    N, C, H, W = x.shape
    x = x.transpose(0, 2, 3, 1).reshape(-1, C)
    xs = np.split(x, G, axis=1)
    x_norms = [None] * G
    means = [None] * G
    variances = [None] * G
    for i in range(G):
        means[i] = np.mean(xs[i], axis=1, keepdims=True)
        variances[i] = np.var(xs[i], axis=1, keepdims=True)
        x_norms[i] = (xs[i] - means[i]) / np.sqrt(variances[i] + eps)

    x_norm = np.concatenate(x_norms, axis=1)
    x_norm = x_norm.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    out = x_norm * gamma + beta

    cache = (xs, means, variances, x_norm, gamma, eps, xs[0].shape[1], G)
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    xs, means, variances, x_norm, gamma, eps, D, G = cache
    N, C, H, W = dout.shape

    dgamma = np.sum(dout*x_norm, axis=(0, 2, 3)).reshape(1, C, 1, 1)
    dbeta = np.sum(dout, axis=(0, 2, 3)).reshape(1, C, 1, 1)
    dx_norms = dout*gamma
    dx_norms = dx_norms.transpose(0, 2, 3, 1).reshape(-1, C)
    dx_norms = np.split(dx_norms, G, axis=1)
    dxs = [None] * G
    for i in range(G):
        dxs[i] = (1 / np.sqrt(variances[i] + eps)) * dx_norms[i]
        dmean = (1 / np.sqrt(variances[i] + eps)) * (-1) * dx_norms[i].sum(axis=1, keepdims=True)
        dvar = ((xs[i] - means[i]) * (-1/2) * (variances[i] + eps)**(-3/2) * dx_norms[i]).sum(axis=1, keepdims=True)
        dmean += (-2/D) * (xs[i] - means[i]).sum(axis=1, keepdims=True) * dvar
        dxs[i] += np.ones_like(xs[i]) * (1/D) * dmean
        dxs[i] += (2/D)*(xs[i] - means[i]) * dvar

    dx = np.concatenate(dxs, axis=1)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    
    return dx, dgamma, dbeta

def softmax_loss(scores, y):
    N = scores.shape[0]

    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    y_probs = probs[range(N), y]
    loss = np.mean(-np.log(y_probs))

    dscores = probs
    dscores[range(N), y] -= 1
    dscores /= N

    return loss, dscores

def svm_loss(scores, y):
    N = scores.shape[0]

    correct_scores = scores[range(N), y].reshape(N, 1)
    margins = np.maximum(0, scores - correct_scores + 1)
    margins[range(N), y] = 0
    loss = np.sum(margins) / N

    dscores = margins > 0
    dscores[range(N), y] = - np.sum(scaler, axis=1)
    dscores /= N

    return loss, dscores


''' Update rules '''

def sgd(x, dx, config):
    config.setdefault('learning_rate', 1e-3)
    x += - config['learning_rate'] * dx
    return x, config

def sgd_momentum(x, dx, config):
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(x))

    v = config['momentum'] * v - config['learning_rate'] * dx
    x += v

    config['velocity'] = v
    return x, config

def nesterov_momentum(x, dx, config):
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(x))

    v_prev = v
    v = config['momentum'] * v - config['learning_rate'] * dx
    x += - config['momentum'] * v_prev + (1 + config['momentum']) * v

    config['velocity'] = v
    return x, config

def adagrad(x, dx, config):
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('epsilon', 1e-8)
    cache = config.get('cache', np.zeros_like(x))

    cache += dx**2
    x += - config['learning_rate'] * dx / (np.sqrt(cache) + config['epsilon'])

    config['cache'] = cache
    return x, config

def rmsprop(x, dx, config):
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    cache = config.get('cache', np.zeros_like(x))

    cache = config['decay_rate'] * cache + (1 - config['decay_rate']) * dx**2
    x += - config['learning_rate'] * dx / (np.sqrt(cache) + config['epsilon'])

    config['cache'] = cache
    return x, config

def adam(x, dx, config):
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    m = config.get('m', np.zeros_like(x))
    v = config.get('v', np.zeros_like(x))
    t = config.get('t', 1)

    m = config['beta1'] * m + (1 - config['beta1']) * dx
    mt = m / (1 - config['beta1']**t)
    v = config['beta2'] * v + (1 - config['beta2']) * dx**2
    vt = v / (1 - config['beta2']**t)
    x += - config['learning_rate'] * mt / (np.sqrt(vt) + config['epsilon'])

    config['m'] = m
    config['v'] = v
    config['t'] = t+1
    return x, config

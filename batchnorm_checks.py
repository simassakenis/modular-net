from modules import batchnorm_forward, batchnorm_backward, batchnorm_backward_alt
from models import FullyConnectedNet
from gradient_check import *
from print_utils import print_formatted, print_mean_std
import numpy as np
import time

def check_batchnorm_forward_train_time():
    print_formatted('Train time batchnorm forward', 'bold', 'blue')

    np.random.seed(231)
    N, D1, D2, D3 = 200, 50, 60, 3
    X = np.random.randn(N, D1)
    W1 = np.random.randn(D1, D2)
    W2 = np.random.randn(D2, D3)
    a = np.maximum(0, X.dot(W1)).dot(W2)

    print('Before batch normalization:')
    print_mean_std(a, axis=0)

    gamma = np.ones((D3,))
    beta = np.zeros((D3,))
    a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
    print('After batch normalization (gamma=1, beta=0)')
    print('(Means should be close to 0 and stds close to 1)')
    print_mean_std(a_norm, axis=0)

    gamma = np.asarray([1.0, 2.0, 3.0])
    beta = np.asarray([11.0, 12.0, 13.0])
    a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
    print('After batch normalization (gamma=', gamma, ', beta=', beta, ')')
    print('(Now means should be close to beta and stds close to gamma)')
    print_mean_std(a_norm, axis=0)


def check_batchnorm_forward_test_time():
    print_formatted('Test time batchnorm forward', 'bold', 'blue')

    np.random.seed(231)
    N, D1, D2, D3 = 200, 50, 60, 3
    W1 = np.random.randn(D1, D2)
    W2 = np.random.randn(D2, D3)

    bn_param = {'mode': 'train'}
    gamma = np.ones(D3)
    beta = np.zeros(D3)

    for t in range(50):
        X = np.random.randn(N, D1)
        a = np.maximum(0, X.dot(W1)).dot(W2)
        batchnorm_forward(a, gamma, beta, bn_param)

    bn_param['mode'] = 'test'
    X = np.random.randn(N, D1)
    a = np.maximum(0, X.dot(W1)).dot(W2)
    a_norm, _ = batchnorm_forward(a, gamma, beta, bn_param)

    print('After batch normalization (test-time):')
    print('(Means should be near 0 and stds near 1)')
    print_mean_std(a_norm, axis=0)


def check_batchnorm_backward():
    print_formatted('Batchnorm backward', 'bold', 'blue')

    np.random.seed(231)
    N, D = 4, 5
    x = 5 * np.random.randn(N, D) + 12
    gamma = np.random.randn(D)
    beta = np.random.randn(D)
    dout = np.random.randn(N, D)

    bn_param = {'mode': 'train'}
    fx = lambda x: batchnorm_forward(x, gamma, beta, bn_param)[0]
    fg = lambda a: batchnorm_forward(x, a, beta, bn_param)[0]
    fb = lambda b: batchnorm_forward(x, gamma, b, bn_param)[0]

    dx_num = evaluate_numerical_gradient_array(fx, x, dout)
    da_num = evaluate_numerical_gradient_array(fg, gamma, dout)
    db_num = evaluate_numerical_gradient_array(fb, beta, dout)

    _, cache = batchnorm_forward(x, gamma, beta, bn_param)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    
    print('(You should expect to see relative errors between 1e-13 and 1e-8)')
    print('dx error: ', relative_error(dx_num, dx))
    print('dgamma error: ', relative_error(da_num, dgamma))
    print('dbeta error: ', relative_error(db_num, dbeta))
    print()


def check_batchnorm_backward_alt():
    print_formatted('Batchnorm backward alt', 'bold', 'blue')

    np.random.seed(231)
    N, D = 100, 500
    x = 5 * np.random.randn(N, D) + 12
    gamma = np.random.randn(D)
    beta = np.random.randn(D)
    dout = np.random.randn(N, D)

    bn_param = {'mode': 'train'}
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)

    t1 = time.time()
    dx1, dgamma1, dbeta1 = batchnorm_backward(dout, cache)
    t2 = time.time()
    dx2, dgamma2, dbeta2 = batchnorm_backward_alt(dout, cache)
    t3 = time.time()

    print('dx difference: ', relative_error(dx1, dx2))
    print('dgamma difference: ', relative_error(dgamma1, dgamma2))
    print('dbeta difference: ', relative_error(dbeta1, dbeta2))
    print('batchnorm_backward_alt is %.2f times faster the batchnorm_backward' % ((t2 - t1) / (t3 - t2)))
    print()


def check_batchnorm_fc_net():
    print_formatted('Fully connected net with batchnorm', 'bold', 'blue')

    np.random.seed(231)
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    print('Relative errors for W should be between 1e-4 ~ 1e-10.')
    print('Relative errors for b should be between 1e-8 ~ 1e-10.')
    print('Relative errors for gammas and betas should be between 1e-8 ~ 1e-9.')
    print()

    for reg in [0, 3.14]:
        print('Running check with reg = ', reg)
        model = FullyConnectedNet(input_dim=D, hidden_dims=[H1, H2], num_classes=C,
                                  weight_scale=5e-2, reg=reg,
                                  normalization='batchnorm')

        loss, grads = model.loss(X, y)
        print('Initial loss: ', loss)

        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = evaluate_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
            print('%s relative error: %.2e' % (name, relative_error(grad_num, grads[name])))

        if reg == 0: print()


def check_batchnorm():
    print_formatted('Batchnorm checks', 'stage')
    check_batchnorm_forward_train_time()
    check_batchnorm_forward_test_time()
    check_batchnorm_backward()
    check_batchnorm_backward_alt()
    check_batchnorm_fc_net()

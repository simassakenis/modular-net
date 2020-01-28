from modules import layernorm_forward, layernorm_backward
from models import FullyConnectedNet
from gradient_check import *
from print_utils import print_formatted, print_mean_std
import numpy as np
import time


def check_layernorm_forward():
    print_formatted('Layernorm forward', 'bold', 'blue')

    np.random.seed(231)
    N, D1, D2, D3 =4, 50, 60, 3
    X = np.random.randn(N, D1)
    W1 = np.random.randn(D1, D2)
    W2 = np.random.randn(D2, D3)
    a = np.maximum(0, X.dot(W1)).dot(W2)

    print('Before layer normalization:')
    print_mean_std(a, axis=1)

    gamma = np.ones(D3)
    beta = np.zeros(D3)
    print('After layer normalization (gamma=1, beta=0)')
    print('(Means should be close to 0 and stds close to 1)')
    a_norm, _ = layernorm_forward(a, gamma, beta)
    print_mean_std(a_norm, axis=1)

    gamma = np.asarray([3.0,3.0,3.0])
    beta = np.asarray([5.0,5.0,5.0])
    print('After layer normalization (gamma=', gamma, ', beta=', beta, ')')
    print('(Now means should be close to beta and stds close to gamma)')
    a_norm, _ = layernorm_forward(a, gamma, beta)
    print_mean_std(a_norm, axis=1)


def check_layernorm_backward():
    print_formatted('Layernorm backward', 'bold', 'blue')

    np.random.seed(231)
    N, D = 4, 5
    x = 5 * np.random.randn(N, D) + 12
    gamma = np.random.randn(D)
    beta = np.random.randn(D)
    dout = np.random.randn(N, D)

    fx = lambda x: layernorm_forward(x, gamma, beta)[0]
    fg = lambda a: layernorm_forward(x, a, beta)[0]
    fb = lambda b: layernorm_forward(x, gamma, b)[0]

    dx_num = evaluate_numerical_gradient_array(fx, x, dout)
    da_num = evaluate_numerical_gradient_array(fg, gamma.copy(), dout)
    db_num = evaluate_numerical_gradient_array(fb, beta.copy(), dout)

    _, cache = layernorm_forward(x, gamma, beta)
    dx, dgamma, dbeta = layernorm_backward(dout, cache)

    print('(You should expect to see relative errors between 1e-12 and 1e-8)')
    print('dx error: ', relative_error(dx_num, dx))
    print('dgamma error: ', relative_error(da_num, dgamma))
    print('dbeta error: ', relative_error(db_num, dbeta))
    print()

def check_layernorm_fc_net():
    print_formatted('Fully connected net with layernorm', 'bold', 'blue')

    np.random.seed(231)
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    print('Relative errors for W should be between 1e-7 ~ 1e-10.')
    print('Relative errors for b should be between 1e-8 ~ 1e-11.')
    print('Relative errors for gammas and betas should be between 1e-6 ~ 1e-10.')
    print()

    for reg in [0, 3.14]:
        print('Running check with reg = ', reg)
        model = FullyConnectedNet(input_dim=D, hidden_dims=[H1, H2], num_classes=C,
                                  weight_scale=5e-2, reg=reg,
                                  normalization='layernorm')

        loss, grads = model.loss(X, y)
        print('Initial loss: ', loss)

        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = evaluate_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
            print('%s relative error: %.2e' % (name, relative_error(grad_num, grads[name])))

        if reg == 0: print()

def check_layernorm():
    print_formatted('Layernorm checks', 'stage')
    check_layernorm_forward()
    check_layernorm_backward()
    check_layernorm_fc_net()

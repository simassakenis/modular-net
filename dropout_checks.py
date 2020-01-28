from modules import dropout_forward, dropout_backward
from models import FullyConnectedNet
from gradient_check import *
from print_utils import print_formatted
import numpy as np

def check_dropout_forward():
    print_formatted('Dropout forward', 'bold', 'blue')

    np.random.seed(231)
    x = np.random.randn(500, 500) + 10

    for p in [0.25, 0.4, 0.7]:
        out, _ = dropout_forward(x, {'mode': 'train', 'p': p})
        out_test, _ = dropout_forward(x, {'mode': 'test', 'p': p})

        print('Running tests with p = ', p)
        print('Mean of input: ', x.mean())
        print('Mean of train-time output: ', out.mean())
        print('Mean of test-time output: ', out_test.mean())
        print('Fraction of train-time output set to zero: ', (out == 0).mean())
        print('Fraction of test-time output set to zero: ', (out_test == 0).mean())
        print()


def check_dropout_backward():
    print_formatted('Dropout backward', 'bold', 'blue')

    np.random.seed(231)
    x = np.random.randn(10, 10) + 10
    dout = np.random.randn(*x.shape)

    dropout_param = {'mode': 'train', 'p': 0.2, 'seed': 123}
    out, cache = dropout_forward(x, dropout_param)
    dx = dropout_backward(dout, cache)
    dx_num = evaluate_numerical_gradient_array(lambda xx: dropout_forward(xx, dropout_param)[0], x, dout)

    print('(Relative error should be around e-10 or less)')
    print('dx relative error: ', relative_error(dx, dx_num))
    print()


def check_dropout_fc_net():
    print_formatted('Fully connected net with dropout', 'bold', 'blue')

    np.random.seed(231)
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    print('Relative errors should be around e-6 or less.')
    print('It is fine if for dropout=1 you have W2 error on the order of e-5.')
    print()

    for dropout in [1, 0.75, 0.5]:
        print('Running check with dropout = ', dropout)
        model = FullyConnectedNet(input_dim=D, hidden_dims=[H1, H2], num_classes=C,
                                  weight_scale=5e-2, dropout=dropout, seed=123)

        loss, grads = model.loss(X, y)
        print('Initial loss: ', loss)

        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = evaluate_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
            print('%s relative error: %.2e' % (name, relative_error(grad_num, grads[name])))

        print()


def check_dropout():
    print_formatted('Dropout checks', 'stage')
    check_dropout_forward()
    check_dropout_backward()
    check_dropout_fc_net()

from modules import conv_forward, conv_backward, max_pool_forward, max_pool_backward
from models import ThreeLayerConvNet
from gradient_check import *
from print_utils import print_formatted
import numpy as np

def check_conv_forward():
    print_formatted('Conv forward', 'bold', 'blue')

    x_shape = (2, 3, 4, 4)
    w_shape = (3, 3, 4, 4)
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
    w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
    b = np.linspace(-0.1, 0.2, num=3)

    conv_param = {'stride': 2, 'pad': 1}
    out, _ = conv_forward(x, w, b, conv_param)
    correct_out = np.array([[[[-0.08759809, -0.10987781],
                               [-0.18387192, -0.2109216 ]],
                              [[ 0.21027089,  0.21661097],
                               [ 0.22847626,  0.23004637]],
                              [[ 0.50813986,  0.54309974],
                               [ 0.64082444,  0.67101435]]],
                             [[[-0.98053589, -1.03143541],
                               [-1.19128892, -1.24695841]],
                              [[ 0.69108355,  0.66880383],
                               [ 0.59480972,  0.56776003]],
                              [[ 2.36270298,  2.36904306],
                               [ 2.38090835,  2.38247847]]]])

    print('Relative error should be around e-8.')
    print('max relative error: ', relative_error(out, correct_out))
    print()


def check_conv_backward():
    print_formatted('Conv backward', 'bold', 'blue')

    np.random.seed(231)
    x = np.random.randn(4, 3, 5, 5)
    w = np.random.randn(2, 3, 3, 3)
    b = np.random.randn(2,)
    dout = np.random.randn(4, 2, 5, 5)
    conv_param = {'stride': 1, 'pad': 1}

    dx_num = evaluate_numerical_gradient_array(lambda x: conv_forward(x, w, b, conv_param)[0], x, dout)
    dw_num = evaluate_numerical_gradient_array(lambda w: conv_forward(x, w, b, conv_param)[0], w, dout)
    db_num = evaluate_numerical_gradient_array(lambda b: conv_forward(x, w, b, conv_param)[0], b, dout)

    out, cache = conv_forward(x, w, b, conv_param)
    dx, dw, db = conv_backward(dout, cache)

    print('Relative errors should be around e-8 or less.')
    print('dx max relative error: ', relative_error(dx, dx_num))
    print('dw max relative error: ', relative_error(dw, dw_num))
    print('db max relative error: ', relative_error(db, db_num))
    print()


def check_max_pool_forward():
    print_formatted('Max pool forward', 'bold', 'blue')

    x_shape = (2, 3, 4, 4)
    x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
    pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}

    out, _ = max_pool_forward(x, pool_param)

    correct_out = np.array([[[[-0.26315789, -0.24842105],
                              [-0.20421053, -0.18947368]],
                             [[-0.14526316, -0.13052632],
                              [-0.08631579, -0.07157895]],
                             [[-0.02736842, -0.01263158],
                              [ 0.03157895,  0.04631579]]],
                            [[[ 0.09052632,  0.10526316],
                              [ 0.14947368,  0.16421053]],
                             [[ 0.20842105,  0.22315789],
                              [ 0.26736842,  0.28210526]],
                             [[ 0.32631579,  0.34105263],
                              [ 0.38526316,  0.4       ]]]])

    print('Relative error should be on the order of e-8.')
    print('max relative error: ', relative_error(out, correct_out))
    print()


def check_max_pool_backward():
    print_formatted('Max pool backward', 'bold', 'blue')

    np.random.seed(231)
    x = np.random.randn(3, 2, 8, 8)
    dout = np.random.randn(3, 2, 4, 4)
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    dx_num = evaluate_numerical_gradient_array(lambda x: max_pool_forward(x, pool_param)[0], x, dout)

    out, cache = max_pool_forward(x, pool_param)
    dx = max_pool_backward(dout, cache)

    print('Relative error should be on the order of e-12.')
    print('dx max relative error: ', relative_error(dx, dx_num))
    print()


def check_conv_net_loss():
    print_formatted('Conv net loss sanity check', 'bold', 'blue')

    model = ThreeLayerConvNet()

    N = 50
    X = np.random.randn(N, 3, 32, 32)
    y = np.random.randint(10, size=N)

    print('num_classes = 10, so initial (softmax) loss should be close to -ln(1/10) = 2.3025850.')
    print('When we use regularization the loss should go up.')

    loss, grads = model.loss(X, y)
    print('Initial loss (no regularization): ', loss)

    model.reg = 0.5
    loss, grads = model.loss(X, y)
    print('Initial loss (with regularization): ', loss)

    print()


def check_conv_net_grads():
    print_formatted('Conv net gradient check', 'bold', 'blue')

    num_inputs = 2
    input_dim = (3, 16, 16)
    num_classes = 10

    np.random.seed(231)
    X = np.random.randn(num_inputs, *input_dim)
    y = np.random.randint(num_classes, size=num_inputs)

    model = ThreeLayerConvNet(input_dim=input_dim, num_filters=3, filter_size=3, hidden_dim=7)
    loss, grads = model.loss(X, y)

    print('(You should expect to see relative errors between 1e-10 and 1e-2)')
    for param_name in sorted(grads):
        f = lambda _: model.loss(X, y)[0]
        param_grad_num = evaluate_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
        print('%s max relative error: %e' % (param_name, relative_error(param_grad_num, grads[param_name])))

    print()


def check_conv():
    print_formatted('Conv checks', 'stage')
    check_conv_forward()
    check_conv_backward()
    check_max_pool_forward()
    check_max_pool_backward()
    check_conv_net_loss()
    check_conv_net_grads()

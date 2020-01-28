from modules import spatial_batchnorm_forward, spatial_batchnorm_backward
from modules import spatial_groupnorm_forward, spatial_groupnorm_backward
from gradient_check import *
from print_utils import print_formatted
import numpy as np


def check_spatial_batchnorm_forward_train_time():
    print_formatted('Train time spatial batchnorm forward', 'bold', 'blue')

    np.random.seed(231)
    N, C, H, W = 2, 3, 4, 5
    x = 4 * np.random.randn(N, C, H, W) + 10

    print('Before spatial batch normalization:')
    print('  Shape: ', x.shape)
    print('  Means: ', x.mean(axis=(0, 2, 3)))
    print('  Stds: ', x.std(axis=(0, 2, 3)))
    print()

    gamma, beta = np.ones(C), np.zeros(C)
    bn_param = {'mode': 'train'}
    out, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)
    print('After spatial batch normalization:')
    print('(Means should be close to 0 and stds close to 1)')
    print('  Shape: ', out.shape)
    print('  Means: ', out.mean(axis=(0, 2, 3)))
    print('  Stds: ', out.std(axis=(0, 2, 3)))
    print()

    gamma, beta = np.asarray([3, 4, 5]), np.asarray([6, 7, 8])
    out, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)
    print('After spatial batch normalization (nontrivial gamma, beta):')
    print('(Means should be close to beta [6, 7, 8] and stds close to gamma [3, 4, 5])')
    print('  Shape: ', out.shape)
    print('  Means: ', out.mean(axis=(0, 2, 3)))
    print('  Stds: ', out.std(axis=(0, 2, 3)))
    print()


def check_spatial_batchnorm_forward_test_time():
    print_formatted('Test time spatial batchnorm forward', 'bold', 'blue')

    np.random.seed(231)
    N, C, H, W = 10, 4, 11, 12
    bn_param = {'mode': 'train'}
    gamma = np.ones(C)
    beta = np.zeros(C)
    for t in range(50):
      x = 2.3 * np.random.randn(N, C, H, W) + 13
      spatial_batchnorm_forward(x, gamma, beta, bn_param)
    bn_param['mode'] = 'test'
    x = 2.3 * np.random.randn(N, C, H, W) + 13
    a_norm, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)

    print('After spatial batch normalization (test-time):')
    print('(Means should be near 0 and stds near 1)')
    print('  means: ', a_norm.mean(axis=(0, 2, 3)))
    print('  stds: ', a_norm.std(axis=(0, 2, 3)))
    print()


def check_spatial_batchnorm_backward():
    print_formatted('Spatial batchnorm backward', 'bold', 'blue')

    np.random.seed(231)
    N, C, H, W = 2, 3, 4, 5
    x = 5 * np.random.randn(N, C, H, W) + 12
    gamma = np.random.randn(C)
    beta = np.random.randn(C)
    dout = np.random.randn(N, C, H, W)

    bn_param = {'mode': 'train'}
    fx = lambda x: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]
    fg = lambda a: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]
    fb = lambda b: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]

    dx_num = evaluate_numerical_gradient_array(fx, x, dout)
    da_num = evaluate_numerical_gradient_array(fg, gamma, dout)
    db_num = evaluate_numerical_gradient_array(fb, beta, dout)

    _, cache = spatial_batchnorm_forward(x, gamma, beta, bn_param)
    dx, dgamma, dbeta = spatial_batchnorm_backward(dout, cache)

    print('(You should expect to see relative errors between 1e-12 and 1e-6)')
    print('dx error: ', relative_error(dx_num, dx))
    print('dgamma error: ', relative_error(da_num, dgamma))
    print('dbeta error: ', relative_error(db_num, dbeta))
    print()


def check_spatial_groupnorm_forward():
    print_formatted('Spatial groupnorm forward', 'bold', 'blue')

    np.random.seed(231)
    N, C, H, W = 2, 6, 4, 5
    G = 2
    x = 4 * np.random.randn(N, C, H, W) + 10
    x_g = x.reshape((N*G,-1))

    print('Before spatial group normalization:')
    print('  Shape: ', x.shape)
    print('  Means: ', x_g.mean(axis=1))
    print('  Stds: ', x_g.std(axis=1))
    print()

    gamma, beta = np.ones((1,C,1,1)), np.zeros((1,C,1,1))
    bn_param = {'mode': 'train'}

    out, _ = spatial_groupnorm_forward(x, gamma, beta, G, bn_param)
    out_g = out.reshape((N*G,-1))
    print('After spatial group normalization:')
    print('(Means should be close to 0 and stds close to 1)')
    print('  Shape: ', out.shape)
    print('  Means: ', out_g.mean(axis=1))
    print('  Stds: ', out_g.std(axis=1))
    print()


def check_spatial_groupnorm_backward():
    print_formatted('Spatial groupnorm backward', 'bold', 'blue')

    np.random.seed(231)
    N, C, H, W = 2, 6, 4, 5
    G = 2
    x = 5 * np.random.randn(N, C, H, W) + 12
    gamma = np.random.randn(1, C, 1, 1)
    beta = np.random.randn(1, C, 1, 1)
    dout = np.random.randn(N, C, H, W)

    gn_param = {}
    fx = lambda x: spatial_groupnorm_forward(x, gamma, beta, G, gn_param)[0]
    fg = lambda a: spatial_groupnorm_forward(x, gamma, beta, G, gn_param)[0]
    fb = lambda b: spatial_groupnorm_forward(x, gamma, beta, G, gn_param)[0]

    dx_num = evaluate_numerical_gradient_array(fx, x, dout)
    da_num = evaluate_numerical_gradient_array(fg, gamma, dout)
    db_num = evaluate_numerical_gradient_array(fb, beta, dout)

    _, cache = spatial_groupnorm_forward(x, gamma, beta, G, gn_param)
    dx, dgamma, dbeta = spatial_groupnorm_backward(dout, cache)

    print('(You should expect to see relative errors between 1e-12 and 1e-7)')
    print('dx error: ', relative_error(dx_num, dx))
    print('dgamma error: ', relative_error(da_num, dgamma))
    print('dbeta error: ', relative_error(db_num, dbeta))
    print()


def check_spatial_norms():
    print_formatted('Check spatial norms', 'stage')
    check_spatial_batchnorm_forward_train_time()
    check_spatial_batchnorm_forward_test_time()
    check_spatial_batchnorm_backward()
    check_spatial_groupnorm_forward()
    check_spatial_groupnorm_backward()

from models import TwoLayerNet, FullyConnectedNet, ThreeLayerConvNet
from solver import Solver
from CIFAR10_data import load_CIFAR10_sample
from plotting import plot_stats
from print_utils import print_formatted
from batchnorm_checks import check_batchnorm
from layernorm_checks import check_layernorm
from dropout_checks import check_dropout
from conv_checks import check_conv
import numpy as np
import pickle
from math import ceil, sqrt
import matplotlib.pyplot as plt
from spatial_norms_checks import check_spatial_norms


''' Hyperparameters '''

subtract_mean = True
normalize = False

''' Data '''

print_formatted('Load data', 'stage')
X_train, y_train, X_val, y_val, X_test, y_test = load_CIFAR10_sample('datasets/cifar-10-batches-py',
                                                                     num_train=49000, num_val=1000, num_test=10000,
                                                                     mean_subtr=subtract_mean, norm=normalize)
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_val shape:', X_val.shape)
print('y_val shape:', y_val.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)


''' Actions '''

def train_two_layer(plot=False):
    print_formatted('Two layer net', 'stage')

    model = TwoLayerNet(input_dim=3072, hidden_dim=100, num_classes=10)
    data = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val}
    solver = Solver(model, data, num_epochs=1, print_every=100, batch_size=100, lr_decay=0.95)
    solver.train()

    if plot: plot_stats('loss', 'train_val_acc', solvers={'two_layer_net': solver}, filename='two_layer_net_stats.png')


def overfit_small_data(plot=False):
    print_formatted('Overfitting small data', 'stage')

    num_train = 50
    small_data = {
      'X_train': X_train[:num_train],
      'y_train': y_train[:num_train],
      'X_val': X_val,
      'y_val': y_val,
    }

    weight_scale = 3e-2
    learning_rate = 1e-3
    update_rule = 'adam'

    model = FullyConnectedNet(input_dim=3072, hidden_dims=[100, 100], num_classes=10, weight_scale=weight_scale)
    solver = Solver(model, small_data,
                    update_rule=update_rule,
                    optim_config={
                      'learning_rate': learning_rate
                    },
                    lr_decay=0.95,
                    num_epochs=20, batch_size=25,
                    print_every=10)
    solver.train()

    if plot: plot_stats('loss', solvers={'fc_net': solver}, filename='overfitting_loss_history.png')


def compare_update_rules(plot=False):
    print_formatted('Update rules', 'stage')

    num_train = 4000
    small_data = {
      'X_train': X_train[:num_train],
      'y_train': y_train[:num_train],
      'X_val': X_val,
      'y_val': y_val,
    }

    learning_rates = {'sgd': 1e-2, 'sgd_momentum': 1e-2, 'nesterov_momentum': 1e-2, 'adagrad': 1e-4, 'rmsprop': 1e-4, 'adam': 1e-3}
    solvers = {}

    for update_rule in ['sgd', 'sgd_momentum', 'nesterov_momentum', 'adagrad', 'rmsprop', 'adam']:
      print_formatted('running with ' + update_rule, 'bold', 'blue')
      model = FullyConnectedNet(input_dim=3072, hidden_dims=[100]*5, num_classes=10, weight_scale=5e-2)

      solver = Solver(model, small_data,
                      num_epochs=5, batch_size=100,
                      update_rule=update_rule,
                      optim_config={
                        'learning_rate': learning_rates[update_rule],
                      },
                      verbose=True)
      solvers[update_rule] = solver
      solver.train()
      print()

    if plot: plot_stats('loss', 'train_acc', 'val_acc', solvers=solvers, filename='update_rules_comparison.png')


def train_with_batchnorm(plot=False):
    print_formatted('Batch normalization', 'stage')

    hidden_dims = [100, 100, 100, 100, 100]
    weight_scale = 2e-2

    num_train = 1000
    small_data = {
      'X_train': X_train[:num_train],
      'y_train': y_train[:num_train],
      'X_val': X_val,
      'y_val': y_val,
    }

    print_formatted('without batchnorm', 'bold', 'blue')
    model = FullyConnectedNet(input_dim=3072, hidden_dims=hidden_dims, num_classes=10, weight_scale=weight_scale)
    solver = Solver(model, small_data,
                    update_rule='adam',
                    optim_config={
                        'learning_rate': 1e-3,
                    },
                    num_epochs=10, batch_size=50,
                    print_every=20)
    solver.train()
    print()

    print_formatted('with batchnorm', 'bold', 'blue')
    bn_model = FullyConnectedNet(input_dim=3072, hidden_dims=hidden_dims, num_classes=10,
                                 weight_scale=weight_scale, normalization='batchnorm')
    bn_solver = Solver(bn_model, small_data,
                       update_rule='adam',
                       optim_config={
                         'learning_rate': 1e-3,
                       },
                       num_epochs=10, batch_size=50,
                       print_every=20)
    bn_solver.train()

    if plot: plot_stats('loss', 'train_acc', 'val_acc', solvers={'baseline': solver, 'with_norm': bn_solver}, filename='batchnorm.png')


def train_with_layernorm(plot=False):
    print_formatted('Layer normalization', 'stage')

    hidden_dims = [100, 100, 100, 100, 100]
    weight_scale = 2e-2

    num_train = 1000
    small_data = {
      'X_train': X_train[:num_train],
      'y_train': y_train[:num_train],
      'X_val': X_val,
      'y_val': y_val,
    }

    print_formatted('without layernorm', 'bold', 'blue')
    model = FullyConnectedNet(input_dim=3072, hidden_dims=hidden_dims, num_classes=10, weight_scale=weight_scale)
    solver = Solver(model, small_data,
                    update_rule='adam',
                    optim_config={
                        'learning_rate': 1e-3,
                    },
                    num_epochs=10, batch_size=50,
                    print_every=20)
    solver.train()
    print()

    print_formatted('with layernorm', 'bold', 'blue')
    ln_model = FullyConnectedNet(input_dim=3072, hidden_dims=hidden_dims, num_classes=10,
                                 weight_scale=weight_scale, normalization='layernorm')
    ln_solver = Solver(ln_model, small_data,
                       update_rule='adam',
                       optim_config={
                         'learning_rate': 1e-3,
                       },
                       num_epochs=10, batch_size=50,
                       print_every=20)
    ln_solver.train()

    if plot: plot_stats('loss', 'train_acc', 'val_acc', solvers={'baseline': solver, 'with_norm': ln_solver}, filename='layernorm.png')


def train_with_dropout(plot=False):
    print_formatted('Dropout', 'stage')

    np.random.seed(231)
    num_train = 500
    small_data = {
      'X_train': X_train[:num_train],
      'y_train': y_train[:num_train],
      'X_val': X_val,
      'y_val': y_val,
    }

    solvers = {}
    dropout_choices = [1, 0.25]
    for dropout in dropout_choices:
        if dropout == 1: print_formatted('without dropout, p = 1', 'bold', 'blue')
        else: print_formatted('with dropout, p = %.2f' % dropout, 'bold', 'blue')

        model = FullyConnectedNet(input_dim=3072, hidden_dims=[500], num_classes=10, dropout=dropout)

        solver = Solver(model, small_data,
                        update_rule='adam',
                        optim_config={
                            'learning_rate': 5e-4,
                        },
                        num_epochs=25, batch_size=100,
                        print_every=100)
        solver.train()
        solvers[dropout] = solver

        if dropout == 1: print()

    if plot: plot_stats('train_acc', 'val_acc', solvers={'1.00 dropout': solvers[1], '0.25 dropout': solvers[0.25]}, filename='dropout.png')


def conv_net_overfitting(plot=False):
    print_formatted('Overfitting small data with convnet', 'stage')

    np.random.seed(231)

    num_train = 100
    small_data = {
        'X_train': X_train[:num_train],
        'y_train': y_train[:num_train],
        'X_val': X_val,
        'y_val': y_val,
    }
    small_data['X_train'] = small_data['X_train'].reshape((small_data['X_train'].shape[0], 32, 32, 3)).transpose(0, 3, 1, 2)
    small_data['X_val'] = small_data['X_val'].reshape((small_data['X_val'].shape[0], 32, 32, 3)).transpose(0, 3, 1, 2)

    model = ThreeLayerConvNet(weight_scale=1e-2)

    solver = Solver(model, small_data,
                    num_epochs=15, batch_size=50,
                    update_rule='adam',
                    optim_config={
                        'learning_rate': 1e-3,
                    },
                    print_every=1)
    solver.train()

    if plot: plot_stats('loss', 'train_val_acc', solvers={'convnet': solver}, filename='convnet_overfitting.png')


def train_conv_net():
    print_formatted('Conv net', 'stage')

    data = {
        'X_train': X_train.reshape((X_train.shape[0], 32, 32, 3)).transpose(0, 3, 1, 2),
        'y_train': y_train,
        'X_val': X_val.reshape((X_val.shape[0], 32, 32, 3)).transpose(0, 3, 1, 2),
        'y_val': y_val,
    }

    model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)

    solver = Solver(model, data,
                    num_epochs=1, batch_size=50,
                    update_rule='adam',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    print_every=20, checkpoint_name='convnet')
    solver.train()


def visualize_convnet_filters():
    print_formatted('Visualizing convnet filters', 'stage')

    checkpoint = pickle.load(open('convnet_epoch_1.pkl', 'rb'))
    W1 = checkpoint['model'].params['W1'].transpose(0, 2, 3, 1)
    N, H, W, C = W1.shape
    grid_size = int(ceil(sqrt(N)))

    for i in range(N):
        img = W1[i]
        low, high = np.min(img), np.max(img)
        rgb_img = 255 * (img - low) / (high - low)
        plt.subplot(grid_size, grid_size, i+1)
        plt.imshow(rgb_img.astype('uint8'))
        plt.axis('off')

    plt.gcf().set_size_inches(10, 10)
    plt.savefig('plots/convnet_filters.png')


def train_best_fc_model(plot=False):
    print_formatted('Best fully connected net', 'stage')

    hidden_dims = [100, 100, 100]
    weight_scale = 2e-2
    num_epochs = 10
    dropout = 1

    data = {
      'X_train': X_train,
      'y_train': y_train,
      'X_val': X_val,
      'y_val': y_val,
      'X_test': X_test,
      'y_test': y_test,
    }

    print_formatted('training', 'bold', 'blue')
    model = FullyConnectedNet(input_dim=3072, hidden_dims=hidden_dims, num_classes=10,
                              weight_scale=weight_scale, normalization='batchnorm', dropout=dropout)
    solver = Solver(model, data,
                    update_rule='adam',
                    optim_config={
                        'learning_rate': 1e-3,
                    },
                    num_epochs=num_epochs, batch_size=50,
                    print_every=100)
    solver.train()
    print()

    if plot: plot_stats('loss', 'train_val_acc', solvers={'best_fc': solver})

    print_formatted('evaluating', 'bold', 'blue')
    y_test_pred = np.argmax(model.loss(data['X_test']), axis=1)
    y_val_pred = np.argmax(model.loss(data['X_val']), axis=1)
    print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
    print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())



# train_two_layer()
# overfit_small_data()
# compare_update_rules()
# check_batchnorm()
# train_with_batchnorm()
# check_layernorm()
# train_with_layernorm(True)
# check_dropout()
# train_with_dropout()
# check_conv()
# conv_net_overfitting()
# train_conv_net()
# visualize_convnet_filters()
# check_spatial_norms()
train_best_fc_model()


print()

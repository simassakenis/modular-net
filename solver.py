import modules
import numpy as np
import pickle

class Solver:

    def __init__(self, model, data, **kwargs):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        self.num_epochs = kwargs.pop('num_epochs', 20)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.verbose = kwargs.pop('verbose', True)
        self.print_every = kwargs.pop('print_every', 10)
        self.checkpoint_name = kwargs.pop('checkpoint_name', None)

        if len(kwargs) > 0:
            extra_args = ', '.join(key for key in kwargs)
            raise ValueError('Unrecognized arguments %s' % extra_args)

        if not hasattr(modules, self.update_rule):
            raise ValueError('No update rule named %s' % self.update_rule)
        self.update_rule = getattr(modules, self.update_rule)

        self._reset()


    def _reset(self):
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.epoch = 1

        self.optim_configs = {}
        for param_name in self.model.params:
            self.optim_configs[param_name] = {key: value for key, value in self.optim_config.items()}


    def _step(self):
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        for param_name, param in self.model.params.items():
            param, self.optim_configs[param_name] = self.update_rule(param, grads[param_name], self.optim_configs[param_name])


    def check_accuracy(self, X, y):
        scores = self.model.loss(X)
        predicted_labels = np.argmax(scores, axis=1)
        accuracy = np.mean(predicted_labels == y)
        return accuracy


    def _save_checkpoint(self):
        if self.checkpoint_name is None: return
        checkpoint = {
            'model': self.model,
            'update_rule': self.update_rule,
            'optim_config': self.optim_config,
            'lr_decay': self.lr_decay,
            'batch_size': self.batch_size,
            'epoch': self.epoch,
            'loss_history': self.loss_history,
            'train_acc_history': self.train_acc_history,
            'val_acc_history': self.val_acc_history,
        }
        filename = self.checkpoint_name + '_epoch_' + str(self.epoch) + '.pkl'
        if self.verbose:
            print('Saving checkpoint to %s' % filename)
        with open(filename, 'wb') as checkpoint_file:
            pickle.dump(checkpoint, checkpoint_file)


    def train(self):
        num_train = self.X_train.shape[0]
        num_iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = num_iterations_per_epoch * self.num_epochs

        for i in range(1, num_iterations+1):
            self._step()

            if i % self.print_every == 0 and self.verbose:
                print('(Iteration %d / %d) loss: %f' % (i, num_iterations, self.loss_history[-1]))

            epoch_end = i % num_iterations_per_epoch == 0
            first_iteration = i == 1
            if first_iteration or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if epoch_end:
                    self._save_checkpoint()
                    if self.verbose: print('(Epoch %d / %d) train_acc: %f, val_acc: %f' % (self.epoch, self.num_epochs, train_acc, val_acc))
                    for param_name in self.model.params:
                        self.optim_configs[param_name]['learning_rate'] *= self.lr_decay
                    self.epoch += 1

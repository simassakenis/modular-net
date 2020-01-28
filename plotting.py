import matplotlib.pyplot as plt


def plot_stats(*stats, solvers={}, filename=None):
    num_subplots = len(stats)
    for i in range(num_subplots):
        plt.subplot(num_subplots, 1, i+1)

        if stats[i] == 'loss':
            plt.title('Training loss')
            for name, solver in solvers.items():
                plt.plot(solver.loss_history, 'o', label=name)
            plt.xlabel('Iteration')
            plt.legend(loc='lower center', ncol=len(solvers))

        elif stats[i] == 'train_acc':
            plt.title('Training Accuracy')
            for name, solver in solvers.items():
                plt.plot(solver.train_acc_history, '-o', label=name)
            plt.xlabel('Epoch')
            plt.legend(loc='lower center', ncol=len(solvers))

        elif stats[i] == 'val_acc':
            plt.title('Validation Accuracy')
            for name, solver in solvers.items():
                plt.plot(solver.val_acc_history, '-o', label=name)
            plt.xlabel('Epoch')
            plt.legend(loc='lower center', ncol=len(solvers))

        elif stats[i] == 'train_val_acc':
            plt.title('Training and Validation Accuracies')
            for name, solver in solvers.items():
                plt.plot(solver.train_acc_history, '-o', label=name+': train')
                plt.plot(solver.val_acc_history, '-o', label=name+': val')
            plt.xlabel('Epoch')
            plt.legend(loc='lower center', ncol=2*len(solvers))

    plt.gcf().set_size_inches(18, 6 * num_subplots)

    if filename is None: plt.show()
    else: plt.savefig('plots/' + filename)

import numpy as np

def evaluate_numerical_gradient(f, x, h=1e-5, verbose=False):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x)
        x[ix] = oldval - h
        fxmh = f(x)
        x[ix] = oldval

        grad[ix] = (fxph - fxmh) / (2 * h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()
    return grad


def evaluate_numerical_gradient_array(f, x, df, h=1e-5):
    # for a 'local' function (line batchnorm) that outputs a numpy array instead of a single loss value
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x)
        x[ix] = oldval - h
        neg = f(x)
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def relative_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def compare_with_numerical_gradient(f, grads_dict, params_dict, verbose=False):
    for param_name in params_dict:
        numerical_gradient = evaluate_numerical_gradient(f, params_dict[param_name], verbose=verbose)
        mean_error = np.mean(np.abs(numerical_gradient - grads_dict[param_name]))
        max_error = np.max(np.abs(numerical_gradient - grads_dict[param_name]))
        rel_error = relative_error(numerical_gradient, grads_dict[param_name])
        print(param_name, 'mean elementwise error:', mean_error)
        print(param_name, 'max elementwise error:', max_error)
        print(param_name, 'rel elementwise error:', rel_error)
        print()

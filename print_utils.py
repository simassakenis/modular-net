import numpy as np

def print_formatted(text, *attributes):
    # example usage: print_formatted('Example text.', 'bold', 'underline', 'green')

    formatting = {
        'bold': '\33[1m',
        'italic': '\33[3m',
        'underline': '\33[4m',
        'blink': '\33[5m',
        'selected': '\33[7m',

        'black': '\33[30m',
        'red': '\33[91m',
        'green': '\33[92m',
        'yellow': '\33[93m',
        'blue': '\33[34m',
        'violet': '\33[35m',
        'beige': '\33[36m',
        'white': '\33[37m',
        'grey': '\33[90m'
    }

    format = ''
    end_format = '\033[0m'
    stage = False
    spaced_above = False
    spaced_below = False

    for attribute in attributes:
        if attribute == 'stage': stage = True
        elif attribute == 'spaced_above': spaced_above = True
        elif attribute == 'spaced_below': spaced_below = True
        elif attribute == 'spaced': spaced_above, spaced_below = True, True
        else: format += formatting[attribute]

    text_to_print = '\n'*spaced_above + format + text + end_format + '\n'*spaced_below

    if stage:
        width = 40
        if len(text) > width-20:
            width = len(text) + 20
        text_to_print = format + '\n' + '*'*width + '\n*****' + '     ' + text.center(width-20) + '     ' +  '*****\n' + '*'*width + '\n' + end_format

    print(text_to_print)

def print_mean_std(x, axis=0):
    print('  means: ', x.mean(axis=axis))
    print('  stds:  ', x.std(axis=axis))
    print()
